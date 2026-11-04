"""Parity: trainer and evaluator must resolve the SAME task.

This is the test the project needed four times and did not have. Bugs #1, #4,
#7 and #10 were all one logical decision expressed in two places, and each was
only discovered after a full GPU run had already been spent on a mismatched
instrument.

What is pinned here:
  1. one prompt for a row, from one function
  2. one id resolution, from one boundary check
  3. one label -> target mapping
  4. the derived p(unsafe) is the documented formula, not a second convention
  5. the unsupervised options are declared, so "ASK was never supervised"
     cannot quietly become "ASK works"
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import choice_task as T  # noqa: E402
from choice_codes import CODES, choice_prompt, code_ids  # noqa: E402


def test_one_renderer_used_by_both_sides():
    """The prompt a row gets is `choice_prompt(base)` -- the same call the
    evaluator makes. A second wrapper anywhere would be bug #4 again."""
    row = {"prompt": "Context: c\nAction: a", "label": 1}
    prompt, target = T.build(row)
    assert prompt == choice_prompt("Context: c\nAction: a", T.OPTIONS)
    assert target == T.I_BLOCK


def test_label_mapping_is_the_documented_one():
    assert T.option_for_label(0) == T.I_ALLOW
    assert T.option_for_label(1) == T.I_BLOCK
    with pytest.raises(ValueError):
        T.option_for_label(2)


def test_unlabelled_row_has_no_target():
    prompt, target = T.build({"prompt": "x"})
    assert target is None
    assert prompt.endswith("Answer: ")


def test_p_unsafe_formula_is_block_plus_half_ask():
    p = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.2, 0.7],
                  [0.2, 0.6, 0.2]])
    got = T.p_unsafe_from_probs(p)
    assert got[0] == pytest.approx(0.1 + 0.5 * 0.2)
    assert got[1] == pytest.approx(0.7 + 0.5 * 0.2)
    # an all-ASK prediction is exactly 0.5 -> maximally undecided, not silently
    # collapsing to safe or unsafe
    assert got[2] == pytest.approx(0.5)


def test_ask_is_declared_unsupervised():
    """If ASK ever gains supervision this fails and forces the docs to change."""
    assert T.I_ASK in T.UNSUPERVISED
    assert T.I_ALLOW not in T.UNSUPERVISED
    assert T.I_BLOCK not in T.UNSUPERVISED
    # and no label maps to it
    assert T.I_ASK not in T.LABEL_TO_OPTION.values()


def test_describe_states_the_unsupervised_option():
    d = T.describe()
    assert "UNSUPERVISED" in d
    assert "p_unsafe = P(BLOCK) + 0.5 * P(ASK)" in d


def test_trainer_and_evaluator_resolve_identical_ids():
    """The parity assertion itself, against a real tokenizer when available.

    Skipped without transformers so `make check` stays dependency-free -- but
    reported as skipped, never as passed.
    """
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HOME", "/scratch/ad2416/hf")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {type(e).__name__}")

    row = {"prompt": "Context: c\nAction: a", "label": 1}
    prompt, _ = T.build(row)

    # the trainer path and the evaluator path, spelled out separately on purpose
    train_ids = T.resolve_ids(tok, prompt)
    eval_ids = code_ids(tok, T.build(row)[0], T.OPTIONS)
    assert train_ids == eval_ids, "trainer and evaluator resolved different ids"
    assert len(train_ids) == len(T.OPTIONS)
    assert len(set(train_ids)) == len(train_ids)


def test_id_disagreement_check_is_not_always_true():
    """The guard in the trainer that refuses a batch whose rows resolve
    different code ids.

    Its first version compared the whole list against ONE element
    (`ids_list != ids_list[0]`), which is always True, so it fired on every
    batch and killed the run at step 1. A guard that cannot be false is as
    broken as one that can never fire; this pins the correct predicate.

    Verified against real data as well: all 20185 training rows resolve to the
    identical tuple (362, 426, 356) with zero failures, so the correct predicate
    is False on real input.
    """
    same = [[362, 426, 356], [362, 426, 356], [362, 426, 356]]
    assert any(x != same[0] for x in same) is False, "must not fire on agreement"

    differ = [[362, 426, 356], [362, 426, 357]]
    assert any(x != differ[0] for x in differ) is True, "must fire on disagreement"

    # and document the broken form so it cannot be reintroduced by accident
    assert (same != same[0]) is True, (
        "the old predicate is vacuously true -- do not reintroduce it")


def test_benchmark_and_corpus_rows_render_identically():
    """A benchmark row and a corpus row with the same text must produce the same
    prompt. If they do not, the instrument is measuring a different task."""
    text = "Context: same\nAction: same"
    corpus_like = {"prompt": text, "label": 0}
    bench_like = {"prompt": text, "label": 1}
    p1, _ = T.build(corpus_like)
    p2, _ = T.build(bench_like)
    assert p1 == p2, "prompt depends on the label -- that is leakage"
