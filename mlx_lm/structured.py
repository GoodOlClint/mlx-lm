"""
Structured-output (JSON-schema / regex / choice) logits processors for mlx-lm.

These processors are designed to compose with mlx-lm's speculative-decoding
and native-MTP paths. The standard ``OutlinesCoreLogitsProcessor`` advances
its internal Outlines ``Guide`` monotonically based on the most-recent
``input_ids[-1]`` it sees — which breaks under speculative/MTP because draft
tokens get fed to the processor *before* the main head verifies them, and
when a draft is rejected mlx-lm rolls back its ``prev_tokens`` array but the
Guide has no corresponding rollback hook.

The processors here treat the ``tokens`` argument they receive on every call
as the source of truth for FSM position. On each call we diff our
already-advanced history against the new ``tokens`` and use
``Guide.rollback_state(n)`` (native to outlines-core) to undo any divergent
suffix, then advance through any newly-committed tokens. This means
structured output stays grammar-correct even when the surrounding
generation loop yields draft tokens that get rejected.

Tradeoff: every accepted draft now pays a Guide ``advance`` per token, and
every rejection pays a ``rollback_state``. For tightly-constrained spans
(e.g. ``Literal["yes","no"]``) acceptance rates can drop because the FSM
limits the draft model's freedom; for loosely-constrained spans (free-form
strings) the speculative speedup is preserved.

Note for users coming from Outlines: ``outlines.from_mlxlm()`` replaces
mlx-lm's generation loop entirely and forfeits speculative decoding and
MTP. The processors here are the alternative — they hook into mlx-lm's
existing ``logits_processors`` parameter and preserve the speculative path.
"""

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Iterable, Optional, Type, Union

import mlx.core as mx
from outlines_core import Guide, Index, outlines_core
from outlines_core.kernels.mlx import (
    allocate_token_bitmask,
    apply_token_bitmask,
    fill_next_token_bitmask,
)


# ---------------------------------------------------------------------------
# Index/Vocabulary cache (shared by all processor classes)
# ---------------------------------------------------------------------------


def _build_outlines_vocabulary(tokenizer):
    """Construct an outlines-core vocabulary object from a HuggingFace tokenizer.

    Reuses the conversion logic from ``outlines.backends.outlines_core`` so
    that an Index compiled here matches one compiled via the Outlines wrapper.
    """
    from outlines.backends.outlines_core import OutlinesCoreBackend

    return OutlinesCoreBackend.create_outlines_core_vocabulary(
        tokenizer.get_vocab(),
        tokenizer.eos_token_id,
        tokenizer.eos_token,
        lambda token: tokenizer.convert_tokens_to_string([token]),
    )


class StructuredProcessorCache:
    """LRU cache of compiled ``outlines_core.Index`` instances, keyed by
    ``(tokenizer fingerprint, regex string)``. Each ``__call__``-style getter
    returns a fresh processor that owns its own ``Guide``; the expensive part
    (regex → DFA compilation) is shared across requests.
    """

    @dataclass(frozen=True)
    class TokenizerFingerprint:
        tokenizer_id: int
        eos_token_id: int
        vocab_size: int

    @dataclass
    class TokenizerCacheEntry:
        index_cache: OrderedDict
        outlines_vocab: Any

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._lock = Lock()
        self._per_tokenizer: Dict[
            StructuredProcessorCache.TokenizerFingerprint,
            StructuredProcessorCache.TokenizerCacheEntry,
        ] = {}

    def _ensure_entry(
        self, tokenizer
    ) -> "StructuredProcessorCache.TokenizerCacheEntry":
        fp = StructuredProcessorCache.TokenizerFingerprint(
            id(tokenizer), tokenizer.eos_token_id, tokenizer.vocab_size
        )
        entry = self._per_tokenizer.get(fp)
        if entry is not None:
            return entry
        entry = StructuredProcessorCache.TokenizerCacheEntry(
            index_cache=OrderedDict(),
            outlines_vocab=_build_outlines_vocabulary(tokenizer),
        )
        self._per_tokenizer[fp] = entry
        return entry

    def get_index_for_regex(self, regex: str, tokenizer) -> Index:
        """Compile (or fetch from cache) an ``Index`` for ``regex``."""
        with self._lock:
            entry = self._ensure_entry(tokenizer)
            cache = entry.index_cache
            if regex in cache:
                index = cache.pop(regex)
                cache[regex] = index
                return index
            index = Index(regex, entry.outlines_vocab)
            cache[regex] = index
            if len(cache) > self.max_size:
                cache.popitem(last=False)
            return index

    # --- back-compat helpers used by mlx_lm.server -------------------------

    def get_processor(self, schema, tokenizer):
        """Return a JSONLogitsProcessor for ``schema`` (or None if schema is None).

        Preserved for callers that only have a JSON schema in hand.
        """
        if schema is None:
            return None
        return JSONLogitsProcessor(schema, tokenizer, cache=self)

    def get_processor_from_model(self, model_cls, tokenizer):
        return JSONLogitsProcessor(model_cls, tokenizer, cache=self)

    def _make_structured_processor(self, schema, tokenizer):
        return self.get_processor(schema, tokenizer)


# ---------------------------------------------------------------------------
# Token-history-driven processor base
# ---------------------------------------------------------------------------


class _RollbackingLogitsProcessor:
    """Base class implementing the token-array-driven Guide synchronization.

    Subclasses provide a compiled ``Index`` via ``__init__`` and inherit the
    ``__call__`` shape contract: ``(tokens: 1D mx.array, logits: (1, V) mx.array)``.

    On the first call, the processor records the input length as the "prompt
    baseline" and biases the first generated token. On subsequent calls it
    diffs the incoming ``tokens`` array against its own advanced-history
    record and uses ``Guide.rollback_state`` to undo any divergent suffix
    before advancing through newly-committed tokens.
    """

    def __init__(self, index: Index, vocab_size: int):
        self._index = index
        self._guide = Guide(index)
        self._vocab_size = vocab_size
        self._bitmask = allocate_token_bitmask(vocab_size)
        # Suffix of `tokens` arg that we have already advanced the Guide through,
        # not including any prompt prefix. Populated lazily after the first call.
        self._advanced: list[int] = []
        self._prompt_len: Optional[int] = None

    def reset(self) -> None:
        """Reset Guide and history so this processor can be reused for a new request."""
        self._guide = Guide(self._index)
        self._advanced = []
        self._prompt_len = None

    @staticmethod
    def _to_int_list(tokens: mx.array) -> list[int]:
        if tokens is None:
            return []
        if not hasattr(tokens, "size") or tokens.size == 0:
            return []
        return [int(t) for t in tokens.tolist()]

    def _sync(self, tokens: mx.array) -> None:
        """Advance/rollback ``self._guide`` so it reflects state after the
        post-prompt suffix of ``tokens``."""
        token_list = self._to_int_list(tokens)
        if self._prompt_len is None:
            self._prompt_len = len(token_list)
            self._advanced = []
            return

        suffix = token_list[self._prompt_len :]

        # Find the longest common prefix of self._advanced and suffix.
        common = 0
        for a, b in zip(self._advanced, suffix):
            if a != b:
                break
            common += 1

        # Roll back any divergent suffix on our side.
        n_rollback = len(self._advanced) - common
        if n_rollback > 0:
            allowed = self._guide.get_allowed_rollback()
            if n_rollback <= allowed:
                self._guide.rollback_state(n_rollback)
            else:
                # Outlines-core's rollback buffer is bounded. Rebuild from
                # scratch by replaying the common prefix on a fresh Guide.
                self._guide = Guide(self._index)
                for tok in suffix[:common]:
                    self._try_advance(tok)
            self._advanced = list(suffix[:common])

        # Advance for any newly-committed tokens. Tokens the Guide doesn't
        # accept (i.e., FSM-invalid drafts proposed by a draft model that
        # doesn't see the bitmask) are skipped silently and NOT recorded in
        # ``self._advanced`` — the verification step at the main head will
        # reject them and mlx-lm will roll back its ``prev_tokens`` array,
        # so the next call's sync will not see them either.
        for tok in suffix[len(self._advanced) :]:
            if self._try_advance(tok):
                self._advanced.append(tok)

    def _try_advance(self, token_id: int) -> bool:
        """Advance Guide by ``token_id`` if accepted; return True iff the
        advance actually happened. Mirrors the issue-#227 guard from
        ``OutlinesCoreLogitsProcessor`` (don't advance past a finished state
        with a non-EOS token) but extends it to ALL non-accepted tokens, so
        FSM-invalid draft tokens don't crash ``Guide.advance``."""
        if not self._guide.accepts_tokens([token_id]):
            return False
        self._guide.advance(token_id, return_tokens=False)
        return True

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        self._sync(tokens)
        # Bias logits using the current Guide state's bitmask.
        # outlines-core's MLX kernel requires (1, V) int32 numpy bitmask and
        # (1, V) MLX logits — exactly what mlx-lm's processor contract delivers.
        fill_next_token_bitmask(self._guide, self._bitmask)
        if logits.ndim == 1:
            biased = apply_token_bitmask(logits[None], self._bitmask)
            return biased.squeeze(0)
        return apply_token_bitmask(logits, self._bitmask)

    @property
    def is_finished(self) -> bool:
        return self._guide.is_finished()


# ---------------------------------------------------------------------------
# Public processor classes
# ---------------------------------------------------------------------------


def _resolve_schema(schema) -> str:
    """Normalize schema input (dict / Pydantic class / JSON string) to a JSON
    string suitable for ``outlines_core.json_schema.build_regex_from_schema``."""
    if isinstance(schema, str):
        try:
            json.loads(schema)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"schema string is not valid JSON: {e}"
            ) from e
        return schema
    if isinstance(schema, dict):
        return json.dumps(schema, sort_keys=True)
    # Pydantic v2 BaseModel subclass.
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        return json.dumps(model_json_schema(), sort_keys=True)
    raise TypeError(
        "schema must be a dict, JSON string, or Pydantic BaseModel subclass; "
        f"got {type(schema).__name__}"
    )


class JSONLogitsProcessor(_RollbackingLogitsProcessor):
    """Constrains generation to JSON conforming to ``schema``.

    Composes correctly with speculative decoding and native MTP: the
    processor uses the ``tokens`` array passed to ``__call__`` as the source
    of truth and rolls back the underlying Outlines ``Guide`` when draft
    tokens are rejected. This pays a small per-token overhead (one
    ``advance`` or ``rollback_state`` call) and may reduce draft-acceptance
    rates on tightly-constrained spans (e.g. enums, integer fields), in
    exchange for grammar correctness end-to-end.

    Args:
        schema: A dict (raw JSON Schema), JSON string, or Pydantic ``BaseModel``
            subclass.
        tokenizer: A HuggingFace-compatible tokenizer; must expose
            ``get_vocab()``, ``eos_token``, ``eos_token_id``,
            ``convert_tokens_to_string``, and ``vocab_size``.
        cache: Optional shared ``StructuredProcessorCache`` for Index reuse
            across requests. Pass the server's cache if you have one;
            otherwise the processor compiles a fresh Index.
    """

    def __init__(
        self,
        schema,
        tokenizer,
        cache: Optional[StructuredProcessorCache] = None,
    ):
        schema_str = _resolve_schema(schema)
        regex = outlines_core.json_schema.build_regex_from_schema(schema_str)
        if cache is None:
            vocab = _build_outlines_vocabulary(tokenizer)
            index = Index(regex, vocab)
        else:
            index = cache.get_index_for_regex(regex, tokenizer)
        super().__init__(index, tokenizer.vocab_size)


class RegexLogitsProcessor(_RollbackingLogitsProcessor):
    """Constrains generation to strings matching ``pattern`` (a regex)."""

    def __init__(
        self,
        pattern: str,
        tokenizer,
        cache: Optional[StructuredProcessorCache] = None,
    ):
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a string, got {type(pattern).__name__}")
        if cache is None:
            vocab = _build_outlines_vocabulary(tokenizer)
            index = Index(pattern, vocab)
        else:
            index = cache.get_index_for_regex(pattern, tokenizer)
        super().__init__(index, tokenizer.vocab_size)


class ChoiceLogitsProcessor(RegexLogitsProcessor):
    """Constrains generation to one of the strings in ``choices``.

    Convenience wrapper that compiles a regex alternation. Useful for
    ``Literal["yes","no"]``-style constraints.
    """

    def __init__(
        self,
        choices: Iterable[str],
        tokenizer,
        cache: Optional[StructuredProcessorCache] = None,
    ):
        choices = list(choices)
        if not choices:
            raise ValueError("choices must be non-empty")
        if not all(isinstance(c, str) for c in choices):
            raise TypeError("all entries in choices must be strings")
        pattern = "(" + "|".join(re.escape(c) for c in choices) + ")"
        super().__init__(pattern, tokenizer, cache=cache)
