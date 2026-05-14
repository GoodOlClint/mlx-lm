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
from typing import Any, Dict, Iterable, Optional, Sequence, Type, Union

import mlx.core as mx
import numpy as np
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


# ---------------------------------------------------------------------------
# Tool-aware JSON-schema processor (composes response_format + tools)
# ---------------------------------------------------------------------------


# Phase labels for ToolAwareJSONLogitsProcessor. Module-level so tests can
# import them without poking at private class attributes.
_PHASE_IDLE = "idle"
_PHASE_OPEN_PREFIX = "open_prefix"
_PHASE_TOOL_BODY = "tool_body"
_PHASE_CLOSE_PREFIX = "close_prefix"
_PHASE_THINK_BODY = "think_body"
_PHASE_THINK_CLOSE_PREFIX = "think_close_prefix"
_PHASE_IN_SCHEMA = "in_schema"
_PHASE_FINISHED = "finished"


class ToolAwareJSONLogitsProcessor:
    """Pass-through-by-default schema processor that composes with tools and thinking.

    Patch 6 design (supersedes Patch 4/5's strict-from-start bitmask):
    schema enforcement is **deferred** until the model voluntarily commits
    to JSON output by emitting ``{``. Before that, the model is free to
    emit anything — preamble text, ``<think>`` blocks, ``<tool_call>``
    blocks, markdown fences — without any bitmask constraint. After
    ``{``, the Outlines ``Guide`` enforces strict schema validity on the
    JSON body. After the JSON body completes, the processor returns to
    pass-through so the model can emit closing markdown fences and stop
    naturally.

    Background — why deferred enforcement (issue #4):
    Patch 4/5 applied a bitmask in ``idle`` (Guide ∪ tool_open ∪
    think_open). Under tool-aware Qwen3.5, this caused the model to
    deterministically pick the minimum-compliant DFA path (empty
    arrays, empty schema shell) under the influence of the chat
    template's tool documentation prompt. Live bisection showed the
    bitmask itself was the cause — a literal no-op processor restored
    natural model behavior (full content, tool calls firing). Patch 6
    achieves the same by making ``idle`` truly pass-through.

    Phases (per generation step):

    * ``idle`` — start of response, or just after a ``</tool_call>`` /
      ``</think>``. **Pass-through.** Watches for one of three
      commitment signals: ``tool_call_start_tokens[0]``,
      ``think_start_tokens[0]``, or the literal ``{`` token. Other
      tokens leave the phase unchanged.
    * ``open_prefix`` — multi-token tool-call open in progress; forces
      the next token of ``tool_call_start_tokens``.
    * ``tool_body`` — pass-through (chat-template's tool grammar
      applies). Watches for ``tool_call_end_tokens[0]``.
    * ``close_prefix`` — multi-token tool-call close in progress;
      pass-through. Returns to ``idle`` on completion, ``tool_body``
      on a false start.
    * ``think_body`` — pass-through. Watches for
      ``think_end_tokens[0]``. The phase machine also *starts* in
      ``think_body`` when the prompt ends inside a ``<think>`` block
      (Qwen3.5/3.6 chat-template auto-injection — see
      ``_detect_initial_phase``).
    * ``think_close_prefix`` — multi-token think close in progress;
      pass-through. Returns to ``idle`` on completion, ``think_body``
      on a false start.
    * ``in_schema`` — strict Guide enforcement. Active from ``{``
      through the final ``}``.
    * ``finished`` — Guide reached its final state. **Pass-through**
      (Patch 6, was: forced EOS in Patch 5) so the model can emit
      closing markdown fences and stop on its own.

    Composes with speculative / MTP decoding: the phase history is
    rebuilt from the ``tokens`` argument on every call. On rollback,
    phase entries are popped and the Guide is rolled back by the count
    of *schema-mode* tokens in the rolled-back suffix (idle / tool /
    think / finished tokens never advanced the Guide).

    Scope: this processor does NOT schema-constrain anything outside the
    ``{...}`` JSON body. The tool-call body's grammar is parser-specific
    (qwen3_coder uses XML-ish ``<function=...><parameter=...>``;
    hermes_json uses JSON); think bodies are free-form natural language.
    The chat template trains the model to produce these correctly.
    """

    def __init__(
        self,
        schema,
        tokenizer,
        tool_call_start_tokens: Optional[Sequence[int]] = None,
        tool_call_end_tokens: Optional[Sequence[int]] = None,
        think_start_tokens: Optional[Sequence[int]] = None,
        think_end_tokens: Optional[Sequence[int]] = None,
        cache: Optional[StructuredProcessorCache] = None,
    ):
        # Tool delimiters (Patch 4) are optional; without them the processor
        # acts as a thinking-aware schema processor only. Think delimiters
        # (Patch 5) are also optional; without them it acts as the
        # tool-aware processor only. With both, it composes tool-calling +
        # thinking + schema. At least one must be provided — with neither,
        # callers should use ``JSONLogitsProcessor`` directly.
        if not tool_call_start_tokens and not think_start_tokens:
            raise ValueError(
                "at least one of tool_call_start_tokens or think_start_tokens "
                "must be a non-empty sequence; use JSONLogitsProcessor for "
                "plain schema-only output"
            )
        self._open_seq = (
            tuple(int(t) for t in tool_call_start_tokens)
            if tool_call_start_tokens
            else ()
        )
        self._close_seq = (
            tuple(int(t) for t in tool_call_end_tokens)
            if tool_call_end_tokens
            else ()
        )
        self._think_open_seq = (
            tuple(int(t) for t in think_start_tokens)
            if think_start_tokens
            else ()
        )
        self._think_close_seq = (
            tuple(int(t) for t in think_end_tokens)
            if think_end_tokens
            else ()
        )

        schema_str = _resolve_schema(schema)
        regex = outlines_core.json_schema.build_regex_from_schema(schema_str)
        if cache is None:
            vocab = _build_outlines_vocabulary(tokenizer)
            self._index = Index(regex, vocab)
        else:
            self._index = cache.get_index_for_regex(regex, tokenizer)
        self._guide = Guide(self._index)
        self._vocab_size = tokenizer.vocab_size
        self._bitmask = allocate_token_bitmask(self._vocab_size)

        # Token suffix we've already stepped the phase machine through,
        # and the corresponding per-position state. Both grow by one entry
        # per accepted post-prompt token. _states[i] is the state AFTER
        # processing _advanced_suffix[i].
        self._advanced_suffix: list[int] = []
        self._states: list[tuple] = []
        self._prompt_len: Optional[int] = None
        # Initial phase is normally IDLE, but Qwen3.5-class chat templates
        # auto-inject ``<think>`` after ``<|im_start|>assistant\n`` so the
        # model is already inside a think block when generation begins. We
        # detect that on the first sync and set this to THINK_BODY so the
        # processor doesn't immediately mask out the model's thinking
        # tokens (issue #2).
        self._initial_phase: str = _PHASE_IDLE
        # The token id that signals "the model is starting JSON output"
        # — i.e., the literal ``{`` for object-root schemas. Patch 6
        # uses this to trigger the transition from pass-through IDLE
        # into Guide-enforced IN_SCHEMA. Pre-compute at construction so
        # we don't have to query the tokenizer per call.
        #
        # Why this matters (issue #4 root cause):
        # When the schema bitmask is active from the first generated
        # token, Qwen3.5 (and similar tool-aware models) get pinned to
        # the minimum-compliant DFA path under the influence of the
        # chat-template's tool documentation — emitting an empty schema
        # shell like ``{"events": []}`` and stopping. Deferring schema
        # enforcement until the model voluntarily commits to JSON
        # (emits ``{``) lets the model write preamble / think / tool-
        # call freely first, then enforces validity on the JSON portion.
        self._json_open_token: Optional[int] = None
        if hasattr(tokenizer, "encode"):
            try:
                json_open_ids = tokenizer.encode("{", add_special_tokens=False)
                if len(json_open_ids) == 1:
                    self._json_open_token = int(json_open_ids[0])
            except (TypeError, ValueError):
                # Some test fixtures or minimal tokenizers may not support
                # the (text, add_special_tokens=...) call shape. Without a
                # known JSON-open token, the processor stays in pass-through
                # IDLE forever; schema enforcement never activates. Callers
                # depending on enforcement should ensure a real tokenizer.
                pass

    # ----- public reset hook ----------------------------------------------

    def reset(self) -> None:
        self._guide = Guide(self._index)
        self._advanced_suffix = []
        self._states = []
        self._prompt_len = None

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _to_int_list(tokens: mx.array) -> list[int]:
        if tokens is None:
            return []
        if not hasattr(tokens, "size") or tokens.size == 0:
            return []
        return [int(t) for t in tokens.tolist()]

    def _current_state(self) -> tuple:
        """The state used as a baseline for the next transition."""
        if self._states:
            return self._states[-1]
        return (self._initial_phase, 0, 0)

    def _detect_initial_phase(self, prompt_tokens: list) -> str:
        """Determine whether the prompt ends inside a ``<think>`` block.

        Some chat templates (Qwen3.5 default) auto-inject ``<think>``
        after ``<|im_start|>assistant\\n``, so the model's first
        generated token is semantically inside a think block. If we don't
        detect this, the IDLE phase's schema-Guide bitmask will mask out
        every non-schema token and the model can't produce reasoning at
        all — the schema gets satisfied inside think and EOS forces an
        empty / minimal content (issue #2).

        Single-token delimiter detection only — Qwen3.5/3.6 have
        ``<think>``/``</think>`` as single special tokens. Multi-token
        delimiters fall back to IDLE conservatively; if the template uses
        them and auto-injects ``<think>``, a small extension to do
        subsequence matching against the prompt tail would be needed.
        """
        if not self._think_open_seq:
            return _PHASE_IDLE
        if len(self._think_open_seq) != 1:
            return _PHASE_IDLE
        if self._think_close_seq and len(self._think_close_seq) != 1:
            return _PHASE_IDLE
        open_id = self._think_open_seq[0]
        close_id = self._think_close_seq[0] if self._think_close_seq else None
        last_open = -1
        last_close = -1
        scan_start = max(0, len(prompt_tokens) - 256)
        for i in range(scan_start, len(prompt_tokens)):
            t = prompt_tokens[i]
            if t == open_id:
                last_open = i
            elif close_id is not None and t == close_id:
                last_close = i
        if last_open > last_close:
            return _PHASE_THINK_BODY
        return _PHASE_IDLE

    def _set_bit(self, token_id: int) -> None:
        """OR a single token bit into ``self._bitmask``."""
        if 0 <= token_id < self._vocab_size:
            self._bitmask[0, token_id >> 5] |= np.int32(1 << (token_id & 31))

    def _force_only(self, token_id: int) -> None:
        """Zero ``self._bitmask`` and set only ``token_id``'s bit."""
        self._bitmask[:] = 0
        self._set_bit(token_id)

    # ----- phase transition (advances Guide as side effect) ---------------

    def _advance_schema(self, t: int, schema_count: int) -> Optional[tuple]:
        """Try to advance the Guide by ``t``; return new state or None if not accepted."""
        if not self._guide.accepts_tokens([t]):
            return None
        self._guide.advance(t, return_tokens=False)
        new_sc = schema_count + 1
        if self._guide.is_finished():
            return (_PHASE_FINISHED, 0, new_sc)
        return (_PHASE_IN_SCHEMA, 0, new_sc)

    def _transition_idle(self, state: tuple, t: int) -> tuple:
        """Patch 6: IDLE is pass-through. Watch for commitment signals.

        The model can emit anything here (preamble like "Here is the
        JSON:", or a leading newline, etc.). We only transition out of
        IDLE when we see an unambiguous commitment to one of three
        modes: tool-call open, think open, or the JSON opening brace.
        Tokens that don't match any signal leave the phase unchanged.
        """
        _, _, sc = state
        if self._open_seq and t == self._open_seq[0]:
            if len(self._open_seq) == 1:
                return (_PHASE_TOOL_BODY, 0, sc)
            return (_PHASE_OPEN_PREFIX, 1, sc)
        if self._think_open_seq and t == self._think_open_seq[0]:
            if len(self._think_open_seq) == 1:
                return (_PHASE_THINK_BODY, 0, sc)
            # Multi-token think_open: treat as committed once first
            # token matches (we don't have prefix-matching for think).
            return (_PHASE_THINK_BODY, 0, sc)
        if self._json_open_token is not None and t == self._json_open_token:
            # The model committed to JSON output. Advance the Guide with
            # ``{`` (which the schema regex always accepts at the initial
            # state for object-root schemas) and enter strict-enforcement
            # mode for the rest of the JSON body.
            if self._guide.accepts_tokens([t]):
                self._guide.advance(t, return_tokens=False)
                new_sc = sc + 1
                if self._guide.is_finished():
                    return (_PHASE_FINISHED, 0, new_sc)
                return (_PHASE_IN_SCHEMA, 0, new_sc)
        # No commitment yet — stay in IDLE pass-through.
        return state

    def _transition_think_body(self, state: tuple, t: int) -> tuple:
        _, _, sc = state
        if self._think_close_seq and t == self._think_close_seq[0]:
            if len(self._think_close_seq) == 1:
                return (_PHASE_IDLE, 0, sc)
            return (_PHASE_THINK_CLOSE_PREFIX, 1, sc)
        return (_PHASE_THINK_BODY, 0, sc)

    def _transition_think_close_prefix(self, state: tuple, t: int) -> tuple:
        _, match_pos, sc = state
        expected = (
            self._think_close_seq[match_pos]
            if match_pos < len(self._think_close_seq)
            else None
        )
        if expected is None or t != expected:
            # Model bailed on closing — fall back into think_body.
            return (_PHASE_THINK_BODY, 0, sc)
        new_mp = match_pos + 1
        if new_mp >= len(self._think_close_seq):
            return (_PHASE_IDLE, 0, sc)
        return (_PHASE_THINK_CLOSE_PREFIX, new_mp, sc)

    def _transition_open_prefix(self, state: tuple, _t: int) -> tuple:
        # We forced this token via the bitmask. Whether the actual token
        # matches or not (e.g., an MTP draft that's about to be rejected),
        # advance the phase machine to what we expect the verified timeline
        # to look like — rollback will fix mismatches.
        _, match_pos, sc = state
        new_mp = match_pos + 1
        if new_mp >= len(self._open_seq):
            return (_PHASE_TOOL_BODY, 0, sc)
        return (_PHASE_OPEN_PREFIX, new_mp, sc)

    def _transition_tool_body(self, state: tuple, t: int) -> tuple:
        _, _, sc = state
        if self._close_seq and t == self._close_seq[0]:
            if len(self._close_seq) == 1:
                return (_PHASE_IDLE, 0, sc)
            return (_PHASE_CLOSE_PREFIX, 1, sc)
        return (_PHASE_TOOL_BODY, 0, sc)

    def _transition_close_prefix(self, state: tuple, t: int) -> tuple:
        _, match_pos, sc = state
        expected = (
            self._close_seq[match_pos] if match_pos < len(self._close_seq) else None
        )
        if expected is None or t != expected:
            # Model bailed on closing — fall back into tool_body.
            return (_PHASE_TOOL_BODY, 0, sc)
        new_mp = match_pos + 1
        if new_mp >= len(self._close_seq):
            return (_PHASE_IDLE, 0, sc)
        return (_PHASE_CLOSE_PREFIX, new_mp, sc)

    def _transition_in_schema(self, state: tuple, t: int) -> tuple:
        _, _, sc = state
        advanced = self._advance_schema(t, sc)
        return advanced if advanced is not None else state

    _TRANSITION_TABLE = {
        _PHASE_IDLE: _transition_idle,
        _PHASE_OPEN_PREFIX: _transition_open_prefix,
        _PHASE_TOOL_BODY: _transition_tool_body,
        _PHASE_CLOSE_PREFIX: _transition_close_prefix,
        _PHASE_THINK_BODY: _transition_think_body,
        _PHASE_THINK_CLOSE_PREFIX: _transition_think_close_prefix,
        _PHASE_IN_SCHEMA: _transition_in_schema,
    }

    def _transition(self, state: tuple, t: int) -> tuple:
        handler = self._TRANSITION_TABLE.get(state[0])
        if handler is None:
            # _PHASE_FINISHED — terminal, no further Guide motion.
            return state
        return handler(self, state, t)

    # ----- sync (rollback + advance) --------------------------------------

    def _rollback_or_rebuild(self, suffix: list, common: int) -> bool:
        """Rewind state to ``common`` post-prompt tokens.

        Returns True if a full rebuild happened (in which case the caller
        does not need to advance — we've already replayed the common
        prefix). Returns False if a simple in-place truncation sufficed.
        """
        old_total_schema = self._current_state()[2]
        new_target_schema = self._states[common - 1][2] if common > 0 else 0
        schema_rollback = old_total_schema - new_target_schema
        if schema_rollback > 0:
            allowed = self._guide.get_allowed_rollback()
            if schema_rollback > allowed:
                # Outlines-core's rollback buffer is bounded. Rebuild by
                # replaying schema-mode tokens through a fresh Guide.
                self._guide = Guide(self._index)
                self._advanced_suffix = []
                self._states = []
                self._advance_new_tail(suffix, 0)
                # The caller's "advance through new tail" is now redundant.
                # Truncate any over-replay (suffix may extend past `common`
                # — that part wasn't in the old history anyway).
                return True
            self._guide.rollback_state(schema_rollback)
        del self._advanced_suffix[common:]
        del self._states[common:]
        return False

    def _sync(self, tokens: mx.array) -> None:
        token_list = self._to_int_list(tokens)
        if self._prompt_len is None:
            self._prompt_len = len(token_list)
            self._advanced_suffix = []
            self._states = []
            self._initial_phase = self._detect_initial_phase(token_list)
            return
        suffix = token_list[self._prompt_len :]

        # Longest common prefix between previous suffix and new suffix.
        common = 0
        for a, b in zip(self._advanced_suffix, suffix):
            if a != b:
                break
            common += 1

        if len(self._advanced_suffix) - common > 0:
            if self._rollback_or_rebuild(suffix, common):
                return

        self._advance_new_tail(suffix, len(self._advanced_suffix))

    def _advance_new_tail(self, suffix: list, start: int) -> None:
        """Advance the phase machine over ``suffix[start:]``."""
        for tok in suffix[start:]:
            new_state = self._transition(self._current_state(), tok)
            # Determine whether this token was "absorbed" into the state.
            # Currently every branch of _transition returns the input
            # state unchanged ONLY when the token was an FSM-invalid
            # candidate in IDLE / IN_SCHEMA (rejected draft). In that
            # case we skip recording so a later rollback's diff doesn't
            # try to undo it. Mirrors _RollbackingLogitsProcessor.
            absorbed = new_state != self._current_state() or self._is_pass_through(
                new_state[0]
            )
            if absorbed:
                self._states.append(new_state)
                self._advanced_suffix.append(tok)

    @staticmethod
    def _is_pass_through(phase: str) -> bool:
        # Patch 6: IDLE and FINISHED are now pass-through too. The only
        # phases that apply a bitmask are OPEN_PREFIX (forces the next
        # token of a multi-token tool_call_start) and IN_SCHEMA (strict
        # Guide enforcement).
        return phase in (
            _PHASE_IDLE,
            _PHASE_TOOL_BODY,
            _PHASE_CLOSE_PREFIX,
            _PHASE_THINK_BODY,
            _PHASE_THINK_CLOSE_PREFIX,
            _PHASE_FINISHED,
        )

    # ----- __call__ -------------------------------------------------------

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        self._sync(tokens)
        phase, match_pos, _ = self._current_state()

        if phase == _PHASE_OPEN_PREFIX:
            # Forced multi-token tool-open sequence. Honor the bitmask.
            self._force_only(self._open_seq[match_pos])
        elif phase == _PHASE_IN_SCHEMA:
            # Strict schema enforcement on the JSON portion of the response.
            fill_next_token_bitmask(self._guide, self._bitmask)
        else:
            # Patch 6: idle, tool_body, close_prefix, think_body,
            # think_close_prefix, finished — all pass-through. The model
            # is free to emit anything (preamble, thinking, tool calls,
            # trailing markdown fences). The wrapper just watches the
            # token stream for transition signals (``<tool_call>``,
            # ``<think>``, ``{``, etc.) via ``_sync`` / ``_transition``.
            #
            # FINISHED used to force EOS via the Guide bitmask. We make
            # it pass-through now so the model can naturally close any
            # surrounding markdown fences or close-think markers after
            # the JSON body completes.
            return logits

        if logits.ndim == 1:
            biased = apply_token_bitmask(logits[None], self._bitmask)
            return biased.squeeze(0)
        return apply_token_bitmask(logits, self._bitmask)

    # ----- introspection (used by tests) ----------------------------------

    @property
    def phase(self) -> str:
        return self._current_state()[0]

    @property
    def is_finished(self) -> bool:
        return self.phase == _PHASE_FINISHED
