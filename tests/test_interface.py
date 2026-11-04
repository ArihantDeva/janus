"""Tests for THE interface: state + question + choices, and nothing else.

The interface is a claim, not a convention: a closed answer set plus state is
enough, and no prose instruction was used. Every test here defends that claim,
because adding "please answer carefully" would quietly invalidate the result
without breaking any code.

Guards verified by reintroducing each defect.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import janus_interface as I  # noqa: E402


# ---------------------------------------------------------------------------
# the contract matches the authoritative upstream shape
# ---------------------------------------------------------------------------

def test_contract_matches_janus_request_shape():
    """{state, questions: {id: {type:'choice', criteria}}} -- verified against
    janus-contracts.ts. If this drifts, the local model and the Janus backend are no
    longer answering the same question and cannot be compared."""
    r = I.as_request({"tool": "bash"})
    assert set(r) == {"state", "questions"}
    assert r["state"] == {"tool": "bash"}
    (qid, q), = r["questions"].items()
    assert qid == I.QUESTION_ID
    assert q["type"] == "choice"
    assert set(q["criteria"]) == set(I.CLASSES)


def test_no_instructions_key_is_ever_emitted():
    """`instructions` is optional upstream and is the one field that would smuggle
    prompt engineering in. It must be absent, not empty."""
    r = I.as_request({"tool": "x"})
    for q in r["questions"].values():
        assert "instructions" not in q


def test_three_classes_fixed_and_ordered():
    """Choice order IS the code order (A,B,C) and the adapter learns those letters.
    Reordering silently relabels every prediction."""
    # The deployed 3-class head is frozen on CLASSES_3 -- that tuple is the
    # eval contract and must never change. CLASSES (4-way) is the Glimmer
    # extension; its first three entries must equal CLASSES_3 in order.
    assert I.CLASSES_3 == ("debug", "fix_input", "fix_call")
    assert I.CLASSES[:3] == I.CLASSES_3
    for i, c in enumerate(I.CLASSES_3):
        assert I.CODES[i] + ")" in I.render({"tool": "t"})
        assert I.choice_index(c) == i
    assert set(I.CLASSES_3) <= set(I.CLASSES)
    assert set(I.CLASSES_3) <= set(I.CRITERIA)


# ---------------------------------------------------------------------------
# rendering is deterministic and complete
# ---------------------------------------------------------------------------

def test_render_is_deterministic():
    """A cached training run must reproduce byte-for-byte, and the code-letter
    boundary check depends on the exact suffix."""
    st = I.build_state(tool="bash", arguments={"a": 1}, error="boom")
    assert I.render(st) == I.render(st)
    assert I.build(st) == I.render(st)


def test_state_contains_all_three_parts_of_the_call():
    """The state is the WHOLE call: tool, arguments and error. v2 stored only the
    error and the tool name, which made `git push` and `pytest` indistinguishable."""
    p = I.render(I.build_state(
        tool="bash", arguments={"command": "pytest -q"}, error="E   AssertionError"))
    assert "tool: bash" in p
    assert "pytest -q" in p
    assert "AssertionError" in p


def test_absent_fields_are_omitted_not_rendered_null():
    """`tool: None` is noise the model must interpret; absence is information."""
    st = I.build_state(error="boom")
    assert "tool" not in st
    assert "arguments" not in st
    p = I.render(st)
    assert "None" not in p and "null" not in p
    assert "error: boom" in p


def test_prompt_ends_at_a_clean_answer_cue():
    """Single trailing space is load bearing: on Llama-3.2-1B a dangling space is a
    standalone token BPE replaces rather than extends, so the code letter would
    become two tokens and the read would score the wrong position."""
    p = I.render({"tool": "t"})
    assert p.endswith("answer: ")
    assert not p.endswith("  ")
    assert I.answer_boundary() in p


def test_arguments_truncate_from_the_end_with_a_marker():
    """A cut argument must be visibly partial. Silent truncation makes the model
    judge a call it was not fully shown."""
    big = {"command": "x" * 20000}
    p = I.render(I.build_state(tool="bash", arguments=big), max_arg_chars=200)
    assert "[truncated]" in p
    assert len(p) < 2000


def test_arguments_that_are_not_json_do_not_crash():
    """toolCall arguments arrive as python-repr strings; a classifier must still
    work on them rather than raising inside a hook."""
    for bad in (None, "", "not json", 42, ["a"], {"k": object()}):
        p = I.render(I.build_state(tool="bash", arguments=bad, error="e"))
        assert "answer: " in p


# ---------------------------------------------------------------------------
# the interface constraint
# ---------------------------------------------------------------------------

def test_clean_prompt_passes_audit():
    I.audit(I.render(I.build_state(tool="bash", error="boom")))


@pytest.mark.parametrize("injection", [
    "You are a senior engineer.",
    "Please classify carefully.",
    "Think step by step.",
    "For example, a timeout means retry.",
    "Always choose debug for tracebacks.",
    "Do not choose fix_call here.",
    "1. read the error",
    "System: be precise",
    "```json```",
    "Take a deep breath and reason carefully.",
])
def test_audit_rejects_prompt_engineering(injection):
    """Each of these would make the result attributable to prose rather than to the
    mechanism. The audit must fire on every one."""
    p = I.render({"tool": "t"}) + injection
    with pytest.raises(ValueError, match="violates"):
        I.audit(p)


def test_audit_passes_the_real_rendered_prompt_for_many_states():
    """The audit must not be so eager it blocks legitimate failure text -- real
    errors contain words like 'never' and 'do not' occasionally, and a false
    positive here would stop training."""
    states = [
        I.build_state(tool="bash", arguments={"command": "rm -rf x"},
                      error="rm: cannot remove 'x': No such file or directory"),
        I.build_state(tool="read", arguments={"path": "/nope"},
                      error="Offset 4000 is beyond end of file (3817 lines total)"),
    ]
    for st in states:
        I.audit(I.render(st))


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def test_decode_picks_a_clear_winner():
    assert I.decode({"debug": 0.7, "fix_input": 0.2, "fix_call": 0.1}) == ("debug", 0.7)


def test_decode_treats_a_tie_as_no_answer():
    """A uniform distribution has no argmax. Taking the first key invents an answer
    -- measured cost: every unmatched benchmark row was labelled `retry` because
    `retry` happened to be first in the dict."""
    for tied in ({c: 1 / 3 for c in I.CLASSES},
                 {"debug": 0.5, "fix_input": 0.5, "fix_call": 0.0}):
        lab, conf = I.decode(tied)
        assert lab is None
        assert conf == pytest.approx(max(tied.values()))


def test_decode_empty():
    assert I.decode({}) == (None, 0.0)


# ---------------------------------------------------------------------------
# the audit's own scope -- a guard that rejects a third of the data is broken
# ---------------------------------------------------------------------------

def test_audit_ignores_instruction_shaped_text_inside_observed_state():
    """THE regression that matters.

    The audit's first version scanned the whole prompt and rejected 1939 of 5014
    real rows (38.67%). Not one of those was prompt engineering -- they were the
    failure text: heredocs and diffs (`<<`), a tool's own wording ("Please provide
    more context to make it unique"), assertion messages ("always"/"never"), and
    pytest's numbered output.

    Rejecting those would silently discard the hardest rows and bias training
    toward clean failures. Observed state is DATA: it may contain anything.
    """
    nasty_but_legitimate = [
        "rm: cannot remove 'x': No such file or directory << heredoc",
        "Found 2 occurrences of edits[0]. Please provide more context to make it unique.",
        "AssertionError: value must always be positive, never negative",
        "1. first test PASSED\n2) second test FAILED",
        "System: something in a log line",
        "e.g. this is inside an error message",
        "Do not use this API, says the deprecation warning",
        "```python\nx = 1\n```",
    ]
    for err in nasty_but_legitimate:
        p = I.render(I.build_state(tool="bash", error=err))
        I.audit(p)          # must NOT raise


def test_audit_still_rejects_scaffolding_that_gains_instructions():
    """Exempting state must not exempt the question or the criteria -- those are
    what we author, and they are where prompt engineering would actually live."""
    p = I.render({"tool": "t"})
    for injected in ("Please be careful.", "You are an expert.",
                     "Think step by step.", "1. read it"):
        patched = p.replace(I.QUESTION, I.QUESTION + " " + injected)
        with pytest.raises(ValueError, match="violates"):
            I.audit(patched)


def test_audit_zero_false_positives_on_real_data():
    """Runs the audit over the actual v3 training set.

    The bound is deliberately 0: any rejection here is a false positive, because
    the scaffolding is generated by this module and cannot contain instructions.
    """
    import json as J
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "responses_v3", "train.jsonl")
    if not os.path.exists(p):
        pytest.skip("v3 dataset not built")
    rejected = 0
    total = 0
    with open(p) as f:
        for line in f:
            r = J.loads(line)
            total += 1
            try:
                I.audit(I.render(I.build_state(
                    tool=r.get("tool"), arguments=r.get("arguments"),
                    error=r.get("error"))))
            except ValueError:
                rejected += 1
    assert total > 1000, "dataset suspiciously small"
    assert rejected == 0, f"{rejected}/{total} legit rows rejected by the audit"
