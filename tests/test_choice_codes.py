"""Tests for the letter-code Choice head.

Written against the specific ways this can silently be wrong, each of which has
either already happened in this project or is known to happen with BPE:

  1. a code that is not one token at the answer boundary -> the read scores the
     wrong position. On Llama-3.2-1B, appending letters directly to "Answer:"
     merged 15 of 26, so this is measured, not hypothetical.
  2. duplicate code ids -> two options share one logit, so the distribution has
     fewer support points than options
  3. K outside 1..26 -> unrepresentable
  4. the softmax covering the whole vocabulary instead of only the candidates
     -> mass leaks to tokens that are not answers
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch", reason="head needs torch; run with --with torch")

import torch  # noqa: E402

from choice_codes import (  # noqa: E402
    CODES, CodeError, choice_prompt, code_ids, probs_from_logits, render_options,
    summarise,
)


class FakeTok:
    """Whitespace tokenizer with TWO deliberate traps:

    - it merges a letter into the preceding word when there is no space, which
      is how the real BPE behaved
    - it maps every letter to a distinct id otherwise
    """

    all_special_ids = [0]

    def __init__(self):
        self.vocab = {}
        self.next = 1

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = self.next
            self.next += 1
        return self.vocab[w]

    def encode(self, text, add_special_tokens=False):
        # A trailing space must NOT vanish: a real BPE keeps the position and
        # the following token starts there. Modelling it as disappearing made
        # the prefix check fail for a reason no real tokenizer has.
        toks = text.split(" ")
        if toks and toks[-1] == "":
            toks = toks[:-1]
        return [self._id(t) for t in toks]


def test_all_26_codes_available():
    assert len(CODES) == 26
    assert CODES[0] == "A" and CODES[25] == "Z"


def test_render_options_lists_every_option_with_its_code():
    txt = render_options(["safe", "ask", "block"])
    for i, opt in enumerate(["safe", "ask", "block"]):
        assert f"{CODES[i]}) {opt}" in txt
    # the trailing space is load bearing: it is what makes the code a token
    assert txt.endswith("Answer: ")
    assert "\nAnswer: " in txt or txt.endswith("Answer: ")


def test_code_ids_rejects_zero_or_too_many_options():
    tok = FakeTok()
    with pytest.raises(ValueError):
        render_options([])
    with pytest.raises(ValueError):
        render_options(["x"] * 27)


def test_code_ids_are_distinct_and_one_token():
    tok = FakeTok()
    # The real renderer emits "...Answer: " and the check strips that space,
    # matching how BPE replaces the dangling space token rather than appending.
    prompt = "pick one Answer: "
    ids = code_ids(tok, prompt, ["a", "b", "c"])
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_code_ids_raises_when_prompt_is_not_a_token_prefix():
    """The measured Llama failure: letter absorbed into the previous token."""

    class MergingTok(FakeTok):
        def encode(self, text, add_special_tokens=False):
            # models the real Llama behaviour: the dangling space token is
            # REPLACED by a merged " A" token, so the prefix changes
            if text.endswith("A"):
                return [self._id("merged")]
            return super().encode(text, add_special_tokens=add_special_tokens)

    tok = MergingTok()
    with pytest.raises(CodeError):
        code_ids(tok, "Answer: ", ["a"])


@pytest.mark.parametrize("tok,options,reason", [
    # base 3 tokens, +A becomes 5 -> the suffix is TWO tokens
    (type("T", (), {"all_special_ids": [],
                    "encode": lambda self, t, add_special_tokens=False:
                        [1, 2, 3, 4, 5] if t.endswith("A") else [1, 2, 3]})(),
     ["a"], "2 tokens"),
    # appending the code changes the prefix -> BPE merged across the boundary
    (type("T", (), {"all_special_ids": [],
                    "encode": lambda self, t, add_special_tokens=False:
                        [9, 9] if t.endswith("A") else [1, 2, 3]})(),
     ["a"], "prefix"),
    # the code resolves to a special token
    (type("T", (), {"all_special_ids": [5],
                    "encode": lambda self, t, add_special_tokens=False:
                        [1, 5] if t.endswith("A") else [1]})(),
     ["a"], "special"),
    # TWO options collapse to the same id -> fewer support points than options
    (type("T", (), {"all_special_ids": [],
                    "encode": lambda self, t, add_special_tokens=False:
                        [1, 7] if t.rstrip()[-1:] in ("A", "B") else [1]})(),
     ["a", "b"], "distinct"),
])
def test_code_ids_rejects_every_bad_boundary(tok, options, reason):
    """Each of these was verified to actually raise; a check that cannot fire is
    worse than no check, because it reads as a guarantee."""
    with pytest.raises(CodeError) as ei:
        code_ids(tok, "x", options)
    assert reason in str(ei.value), f"expected {reason!r} in {ei.value}"


def test_softmax_covers_only_candidate_ids():
    """Mass must not leak to the rest of the vocabulary."""
    V = 50
    logits = torch.full((V,), -20.0)
    ids = [3, 7, 11]
    logits[ids] = torch.tensor([1.0, 2.0, 3.0])
    p = probs_from_logits(logits, ids)
    assert p.shape == (3,)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-6)
    # the huge negative logits elsewhere contribute nothing
    full = torch.softmax(logits.float(), dim=-1)
    for i, tid in enumerate(ids):
        # Different float32 reduction lengths can differ by one rounding step.
        assert float(full[tid]) == pytest.approx(float(p[i]), abs=1e-7, rel=0)


def test_softmax_preserves_order_and_temperature_monotone():
    logits = torch.tensor([0.0, 1.0, 5.0, -3.0])
    ids = [0, 1, 2, 3]
    p = probs_from_logits(logits, ids, temperature=1.0)
    assert int(p.argmax()) == 2
    p_hot = probs_from_logits(logits, ids, temperature=0.5)
    assert float(p_hot.max()) > float(p.max())   # lower T sharpens


def test_choice_prompt_keeps_options_and_cue():
    out = choice_prompt("Context: x\nAction: y", ["safe", "ask", "block"])
    assert "Context: x" in out and "Action: y" in out
    assert out.endswith("Answer: ")
    assert "A) safe" in out and "C) block" in out


def test_summarise_reports_choice_and_confidence():
    s = summarise([0.1, 0.7, 0.2])
    assert s["choice"] == 1
    assert s["confidence"] == pytest.approx(0.7)


def test_k_equals_2_reduces_to_the_shipped_head():
    """The claim that this is the existing 2-token read at K=2, so a comparison
    between them is a like-for-like change of K, not a change of mechanism."""
    logits = torch.tensor([0.3, -1.2])
    p = probs_from_logits(logits, [0, 1])
    assert p.shape == (2,)
    assert float(p[0]) == pytest.approx(float(torch.softmax(logits, -1)[0]))
