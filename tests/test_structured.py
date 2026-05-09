"""Tests for mlx_lm.structured logits processors.

These split into two groups:

1. Pure-Python tests of the token-history-driven sync logic in
   ``_RollbackingLogitsProcessor``. No model is loaded; a tiny fake
   tokenizer is wired directly into ``outlines_core.Index``. These verify
   that:
     - ``advance`` and ``rollback_state`` are called the right number of
       times under standard, MTP-style, and external-draft-style call
       patterns.
     - FSM-invalid draft tokens are skipped without raising and are not
       tracked, so they don't poison subsequent sync attempts.
     - The bitmask correctly disallows invalid tokens.

2. Integration tests that run ``mtp_generate_step`` and
   ``speculative_generate_step`` end-to-end with a structured processor
   active, asserting the generated tokens form a sequence the regex DFA
   accepts. These use the same tiny synthetic Qwen3.5 model as
   ``tests/test_mtp.py`` so they run in CI time without HF downloads.
"""

import importlib
import json
import unittest

import mlx.core as mx

from mlx_lm.structured import (
    ChoiceLogitsProcessor,
    JSONLogitsProcessor,
    RegexLogitsProcessor,
    StructuredProcessorCache,
    _RollbackingLogitsProcessor,
    _resolve_schema,
)


# ---------------------------------------------------------------------------
# Fake tokenizer for sync-logic tests
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """A minimal stand-in matching the surface ``outlines_core`` requires.

    Vocabulary maps single-character tokens 'a'..'d' to ids 0..3, plus
    ``<eos>`` (id 4) and ``<pad>`` (id 5). Just enough to compile a small
    regex into an Index.
    """

    def __init__(self):
        self._vocab = {chr(ord("a") + i): i for i in range(4)}
        self._vocab["<eos>"] = 4
        self._vocab["<pad>"] = 5
        self.eos_token_id = 4
        self.eos_token = "<eos>"
        self.vocab_size = 6

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_string(self, toks):
        return "".join(toks)


class _JSONFriendlyTokenizer:
    """Each printable ASCII character is its own token id, plus an EOS slot.

    Outlines-core requires every literal character used in the regex to have
    at least one single-token representation in the vocabulary; this builds
    the simplest such vocab (256 single-char tokens).
    """

    def __init__(self):
        self._strs = [chr(i) for i in range(256)]
        # Outlines-core requires unique token strings. ASCII 0..31 and 128+
        # are non-printable but chr(i) still returns distinct strings.
        self.eos_token_id = 0
        self.eos_token = self._strs[0]
        self.vocab_size = 256
        self._vocab = {s: i for i, s in enumerate(self._strs)}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_string(self, toks):
        return "".join(toks)


def _make_index(regex: str, tokenizer: FakeTokenizer):
    from outlines.backends.outlines_core import OutlinesCoreBackend
    from outlines_core import Index

    vocab = OutlinesCoreBackend.create_outlines_core_vocabulary(
        tokenizer.get_vocab(),
        tokenizer.eos_token_id,
        tokenizer.eos_token,
        lambda t: tokenizer.convert_tokens_to_string([t]),
    )
    return Index(regex, vocab)


# ---------------------------------------------------------------------------
# Sync-logic unit tests (no model)
# ---------------------------------------------------------------------------


class TestRollbackingProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FakeTokenizer()
        cls.regex = r"a{1,3}"
        cls.index = _make_index(cls.regex, cls.tokenizer)

    def setUp(self):
        self.proc = _RollbackingLogitsProcessor(self.index, self.tokenizer.vocab_size)
        self.logits = mx.zeros((1, self.tokenizer.vocab_size))

    def test_first_call_records_prompt_baseline(self):
        """First call sets prompt_len and does not advance the Guide."""
        self.proc(mx.array([5], dtype=mx.int32), self.logits)
        self.assertEqual(self.proc._advanced, [])
        self.assertEqual(self.proc._prompt_len, 1)

    def test_linear_generation_advances_per_token(self):
        """Standard (non-speculative) generation advances Guide once per call."""
        self.proc(mx.array([5], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0], dtype=mx.int32), self.logits)
        self.assertEqual(self.proc._advanced, [0])
        self.proc(mx.array([5, 0, 0], dtype=mx.int32), self.logits)
        self.assertEqual(self.proc._advanced, [0, 0])
        self.proc(mx.array([5, 0, 0, 0], dtype=mx.int32), self.logits)
        self.assertEqual(self.proc._advanced, [0, 0, 0])
        # Three a's saturates the {1,3} quantifier; Guide should be in a final state.
        self.assertTrue(self.proc._guide.is_finished())

    def test_invalid_draft_skipped_not_tracked(self):
        """A draft token the FSM doesn't accept must not crash and must not
        appear in self._advanced (so subsequent rollback bookkeeping is correct)."""
        self.proc(mx.array([5], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0], dtype=mx.int32), self.logits)
        # 'b' (id=1) is not in the regex r"a{1,3}".
        self.proc(mx.array([5, 0, 1], dtype=mx.int32), self.logits)
        self.assertEqual(self.proc._advanced, [0])

    def test_rollback_after_invalid_draft(self):
        """After an FSM-invalid draft is rejected and prev_tokens shrinks,
        the next call's Guide state should match a fresh advance through the
        committed prefix."""
        self.proc(mx.array([5], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0, 1], dtype=mx.int32), self.logits)  # invalid draft
        self.proc(mx.array([5, 0, 0], dtype=mx.int32), self.logits)  # verify_pred='a'
        self.assertEqual(self.proc._advanced, [0, 0])

    def test_rollback_after_valid_but_rejected_draft(self):
        """When a draft is FSM-valid but rejected (verify_pred didn't match),
        the Guide must roll back exactly one step before advancing for verify_pred."""
        self.proc(mx.array([5], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0], dtype=mx.int32), self.logits)
        # Valid draft — Guide tracks it.
        self.proc(mx.array([5, 0, 0], dtype=mx.int32), self.logits)
        self.assertEqual(self.proc._advanced, [0, 0])
        # Rejected (e.g., main head sampled 'a' at slot 1 but draft was 'a' too —
        # contrived: imagine main sampled '<eos>' instead). prev_tokens rolls back
        # by one and the next call sees the verify_pred token.
        self.proc(mx.array([5, 0, 4], dtype=mx.int32), self.logits)
        # eos isn't accepted at this state in outlines-core (it's not part of
        # the regex DFA's transition set), so it's skipped. Guide is at the
        # state after one 'a' — _advanced reflects the rollback.
        self.assertEqual(self.proc._advanced, [0])

    def test_rollback_uses_native_guide_api(self):
        """The processor should call Guide.rollback_state when the rollback
        depth is within the buffer, not rebuild from scratch."""
        self.proc(mx.array([5], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0], dtype=mx.int32), self.logits)
        # Track the Guide identity; rollback should reuse it (not replace).
        guide_id = id(self.proc._guide)
        self.proc(mx.array([5, 0, 0], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0], dtype=mx.int32), self.logits)
        self.assertEqual(id(self.proc._guide), guide_id)

    def test_bitmask_disallows_invalid_first_token(self):
        """At the regex's start state, only 'a' should have a finite logit."""
        out = self.proc(mx.array([5], dtype=mx.int32), self.logits)
        out_np = out.tolist()[0]
        self.assertEqual(out_np[0], 0.0)  # 'a' allowed
        for i in (1, 2, 3, 4, 5):
            self.assertEqual(out_np[i], float("-inf"))

    def test_bitmask_recovers_after_rollback(self):
        """A bitmask query after rollback should equal the bitmask at the
        equivalent point of a fresh-advance trajectory."""
        # Trajectory A: linear advance through one 'a'.
        proc_a = _RollbackingLogitsProcessor(self.index, self.tokenizer.vocab_size)
        proc_a(mx.array([5], dtype=mx.int32), self.logits)
        out_a = proc_a(mx.array([5, 0], dtype=mx.int32), self.logits)

        # Trajectory B: advance through 'a', accept invalid draft, get rolled back.
        proc_b = _RollbackingLogitsProcessor(self.index, self.tokenizer.vocab_size)
        proc_b(mx.array([5], dtype=mx.int32), self.logits)
        proc_b(mx.array([5, 0], dtype=mx.int32), self.logits)
        proc_b(mx.array([5, 0, 0], dtype=mx.int32), self.logits)
        out_b = proc_b(mx.array([5, 0], dtype=mx.int32), self.logits)
        self.assertEqual(out_a.tolist(), out_b.tolist())

    def test_reset_clears_history(self):
        self.proc(mx.array([5], dtype=mx.int32), self.logits)
        self.proc(mx.array([5, 0], dtype=mx.int32), self.logits)
        self.proc.reset()
        self.assertEqual(self.proc._advanced, [])
        self.assertIsNone(self.proc._prompt_len)


# ---------------------------------------------------------------------------
# Public processor classes
# ---------------------------------------------------------------------------


class TestPublicProcessors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FakeTokenizer()
        cls.cache = StructuredProcessorCache()

    def test_regex_processor(self):
        proc = RegexLogitsProcessor(r"a{1,3}", self.tokenizer, cache=self.cache)
        self.assertIsInstance(proc, _RollbackingLogitsProcessor)

    def test_choice_processor_normalizes_to_alternation(self):
        proc = ChoiceLogitsProcessor(["a", "b"], self.tokenizer, cache=self.cache)
        self.assertIsInstance(proc, _RollbackingLogitsProcessor)

    def test_choice_processor_rejects_empty(self):
        with self.assertRaises(ValueError):
            ChoiceLogitsProcessor([], self.tokenizer)

    def test_json_processor_dict_schema(self):
        # The schema's compiled regex uses literal '{', '}', ':', '"', '0'-'9'
        # characters. Build a vocab that gives each of those a single-token
        # representation so outlines-core can find DFA transitions for them.
        proc = JSONLogitsProcessor(
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            _JSONFriendlyTokenizer(),
        )
        self.assertIsInstance(proc, _RollbackingLogitsProcessor)

    def test_resolve_schema_dict(self):
        s = _resolve_schema({"type": "string"})
        self.assertEqual(json.loads(s), {"type": "string"})

    def test_resolve_schema_string(self):
        self.assertEqual(_resolve_schema('{"type":"string"}'), '{"type":"string"}')

    def test_resolve_schema_pydantic(self):
        from pydantic import BaseModel

        class M(BaseModel):
            x: int

        s = _resolve_schema(M)
        self.assertIn("integer", s)

    def test_resolve_schema_invalid_string(self):
        with self.assertRaises(ValueError):
            _resolve_schema("not json")

    def test_resolve_schema_invalid_type(self):
        with self.assertRaises(TypeError):
            _resolve_schema(42)

    def test_cache_reuses_index_for_same_regex(self):
        cache = StructuredProcessorCache()
        idx_a = cache.get_index_for_regex(r"a+", self.tokenizer)
        idx_b = cache.get_index_for_regex(r"a+", self.tokenizer)
        self.assertIs(idx_a, idx_b)


# ---------------------------------------------------------------------------
# Integration tests with the synthetic Qwen3.5 MTP model
# ---------------------------------------------------------------------------


def _make_qwen3_5_mtp_model():
    """Same tiny model fixture as ``tests/test_mtp.py``."""
    module = importlib.import_module("mlx_lm.models.qwen3_5")
    args = module.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 32,
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
            },
        }
    )
    model = module.Model(args)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


class _ToyTokenizer:
    """Minimal HF-tokenizer-like surface for the 256-vocab synthetic model.

    Each token is an ASCII char (or '?' for non-printable). Outlines-core
    will compile a regex over this alphabet; the model's logits are random
    so we test that whatever the model picks remains FSM-valid, not that it
    picks anything in particular.
    """

    def __init__(self, vocab_size: int = 256, eos_id: int = 1):
        self._strs = [chr(i) if 32 <= i < 127 else "?" for i in range(vocab_size)]
        # Make every string unique by suffixing the id — outlines-core requires
        # the vocab to be a bijection.
        self._strs = [f"{s}<{i}>" for i, s in enumerate(self._strs)]
        self.eos_token_id = eos_id
        self.eos_token = self._strs[eos_id]
        self.vocab_size = vocab_size
        self._vocab = {s: i for i, s in enumerate(self._strs)}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_string(self, toks):
        return "".join(toks)


class TestStructuredWithMTP(unittest.TestCase):
    """Confirms structured output composes with native MTP without crashing
    and produces FSM-valid sequences."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_qwen3_5_mtp_model()
        cls.tokenizer = _ToyTokenizer(vocab_size=256, eos_id=1)
        # A regex that allows any sequence of "letter<id>" tokens up to length 8,
        # phrased so nearly every vocab token matches at every position. This
        # exercises the FSM machinery (advance/rollback) under MTP without
        # over-constraining the random-weight model.
        cls.regex = r".{1,32}"
        cls.proc_factory = lambda self: RegexLogitsProcessor(
            r".{1,32}", self.tokenizer
        )

    def test_mtp_with_structured_does_not_crash(self):
        from mlx_lm.generate import mtp_generate_step

        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        proc = self.proc_factory()
        toks = []
        for tok, _, _ in mtp_generate_step(
            prompt, self.model, max_tokens=8, logits_processors=[proc]
        ):
            toks.append(int(tok))
            if len(toks) >= 8:
                break
        self.assertEqual(len(toks), 8)

    def test_mtp_with_structured_emits_fsm_valid_sequence(self):
        """A fresh Guide replayed through the committed token output should
        accept every non-EOS token without raising — i.e., the surrounding
        generation loop only ever yielded tokens permitted by the regex DFA.

        EOS is excluded from the replay: outlines-core marks EOS as bitmask-
        permitted at final states (so generation can stop) but
        ``Guide.accepts_tokens([eos])`` returns False, so ``advance(eos)``
        would raise even though emitting EOS is correct behavior."""
        from outlines_core import Guide

        from mlx_lm.generate import mtp_generate_step

        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        proc = self.proc_factory()
        toks = []
        for tok, _, _ in mtp_generate_step(
            prompt, self.model, max_tokens=8, logits_processors=[proc]
        ):
            toks.append(int(tok))
            if len(toks) >= 8:
                break
        guide = Guide(proc._index)
        eos_id = self.tokenizer.eos_token_id
        for step, t in enumerate(toks):
            if t == eos_id:
                continue
            if guide.accepts_tokens([t]):
                guide.advance(t, return_tokens=False)
            else:
                self.fail(
                    f"Yielded FSM-invalid token {t} at step {step}; "
                    f"prefix so far = {toks[:step]}"
                )


def _make_llama_model():
    """Tiny pure-attention Llama. The Qwen3.5 fixture used elsewhere has
    SSM (ArraysCache) layers, which the external-draft
    ``speculative_generate_step`` rejects via its ``can_trim_prompt_cache``
    check. Llama uses only KVCache so it works under that path."""
    module = importlib.import_module("mlx_lm.models.llama")
    args = module.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=256,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = module.Model(args)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


class TestStructuredWithSpeculative(unittest.TestCase):
    """Same composition test for the external-draft speculative path."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_llama_model()
        cls.draft = _make_llama_model()
        cls.tokenizer = _ToyTokenizer(vocab_size=256, eos_id=1)

    def test_speculative_with_structured_does_not_crash(self):
        from mlx_lm.generate import speculative_generate_step

        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        proc = RegexLogitsProcessor(r".{1,32}", self.tokenizer)
        toks = []
        for tok, _, _ in speculative_generate_step(
            prompt,
            self.model,
            self.draft,
            num_draft_tokens=2,
            max_tokens=8,
            logits_processors=[proc],
        ):
            toks.append(int(tok))
            if len(toks) >= 8:
                break
        self.assertEqual(len(toks), 8)


class TestChoiceConstraint(unittest.TestCase):
    """Negative test: ChoiceLogitsProcessor must produce only the listed
    choices, even with a randomly-initialized model."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_qwen3_5_mtp_model()
        # Build a tokenizer whose vocab includes specific multi-character
        # tokens "yes" and "no" so the choice constraint maps to single tokens.
        cls.tokenizer = _ToyTokenizer(vocab_size=256, eos_id=1)
        # Override two specific token strings to be unambiguous choices.
        cls.tokenizer._strs[42] = "yes"
        cls.tokenizer._strs[43] = "no"
        cls.tokenizer._vocab = {s: i for i, s in enumerate(cls.tokenizer._strs)}

    def test_choice_constrains_first_token(self):
        from mlx_lm.generate import generate_step

        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        proc = ChoiceLogitsProcessor(["yes", "no"], self.tokenizer)
        first_tok, _ = next(
            iter(
                generate_step(
                    prompt,
                    self.model,
                    max_tokens=1,
                    logits_processors=[proc],
                )
            )
        )
        # Only the two choice token ids should be reachable from the start state.
        self.assertIn(int(first_tok), (42, 43))


if __name__ == "__main__":
    unittest.main()
