"""Public hook boundary: stdout is JSON; model input matches training."""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import backends as B
import muse_hook as H

PAYLOAD = {"tool_name": "bash", "tool_input": {"command": "pytest -q"},
           "error": "E AssertionError: 1 != 2"}

SKIP_TOOLS = {"submit_reminder_decision"}


class FixedBackend:
    name = "keyword"

    def __init__(self, dist=None, noisy=False):
        self.dist = dist or {"debug": 0.9, "fix_input": 0.07, "fix_call": 0.03}
        self.noisy = noisy

    def __call__(self, text):
        if self.noisy:
            print("inference diagnostic")
        return self.dist


class EncoderContract(FixedBackend):
    name = "janus"

    def __call__(self, text):
        prefix = ('state\ntool: bash\narguments: {"command": "pytest -q"}\n'
                  'error: E AssertionError: 1 != 2\n\nquestion\n')
        if not text.startswith(prefix) or "\n\nchoices\nA) debug:" not in text:
            raise ValueError("not the training interface")
        if "answer:" in text:
            raise ValueError("encoder has no answer cue")
        return self.dist


def invoke(monkeypatch, factory, payload=PAYLOAD, threshold="0.60"):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(B, "get_backend", factory)
    monkeypatch.setenv("JANUS_HOOK_DISABLE", "")
    monkeypatch.setenv("JANUS_THRESHOLD", threshold)
    return H.main()


@pytest.mark.parametrize("phase", ["load", "inference"])
def test_backend_diagnostics_cannot_pollute_hook_stdout(monkeypatch, capsys, phase):
    """Removing the diagnostic redirection must make json.loads fail."""
    def factory():
        if phase == "load":
            print("causal masks: 16 before -> 0 after")
        return FixedBackend(noisy=phase == "inference")

    assert invoke(monkeypatch, factory) == 0
    out = capsys.readouterr()
    context = json.loads(out.out)["hookSpecificOutput"]["additionalContext"]
    assert "debug" in context and "[janus] keyword read this as debug" in context
    assert "diagnostic" in out.err or "causal masks" in out.err


def test_janus_receives_state_question_choices_not_plain_failure_text(monkeypatch,
                                                                       capsys):
    """Using failure_text for Janus must lose the advisory, not pass."""
    assert invoke(monkeypatch, EncoderContract) == 0
    out = json.loads(capsys.readouterr().out)
    text = out["hookSpecificOutput"]["additionalContext"]
    assert "debug" in text and "[janus] janus read this as debug" in text


def test_classifier_failure_is_visible_without_leaking_exception_details(monkeypatch,
                                                                        capsys):
    def unavailable():
        raise RuntimeError("sensitive provider detail")

    assert invoke(monkeypatch, unavailable) == 0
    out = capsys.readouterr()
    assert out.out == ""
    assert "RuntimeError" in out.err
    assert "sensitive provider detail" not in out.err


@pytest.mark.parametrize("threshold", ["bad", "nan", "inf", "-0.1", "1.1"])
def test_invalid_threshold_is_inert_instead_of_crashing_or_advising(monkeypatch,
                                                                  capsys, threshold):
    def forbidden():
        pytest.fail("invalid configuration must not construct a classifier")

    assert invoke(monkeypatch, forbidden, threshold=threshold) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("confidence", [0.55, 0.9])
def test_compact_mode_renders_inline_verdict(monkeypatch, capsys, confidence):
    """JANUS_COMPACT=1: single-line verdict, no [janus saw] block."""
    dist = {"debug": confidence, "fix_input": 1 - confidence, "fix_call": 0.0}
    monkeypatch.setenv("JANUS_COMPACT", "1")
    assert invoke(monkeypatch, lambda: FixedBackend(dist)) == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    if confidence >= 0.60:
        assert context.splitlines()[0].startswith("[janus")
        assert "debug" in context.splitlines()[0]
    else:
        assert "unsure" in context
    assert "[janus saw]" not in context


def test_both_channels_present_by_default(monkeypatch, capsys):
    """Default LOUD: additionalContext has state block AND systemMessage has one-liner."""
    assert invoke(monkeypatch, EncoderContract) == 0
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert "[janus saw]" in out["additionalContext"]
    assert out["systemMessage"].startswith("[janus")
    assert "[janus saw]" not in out["systemMessage"]


@pytest.mark.parametrize("confidence", [0.55, 0.9])
def test_both_advisory_branches_name_response_class_and_classifier(monkeypatch,
                                                                  capsys, confidence):
    dist = {"debug": confidence, "fix_input": 1 - confidence, "fix_call": 0.0}
    assert invoke(monkeypatch, lambda: FixedBackend(dist)) == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "debug" in context and "keyword" in context
    assert ("[janus] keyword read this as" in context) == (confidence >= 0.60)
    assert ("unsure" in context) == (confidence < 0.60)


def test_tied_distribution_never_invents_a_class_even_below_default_threshold(
        monkeypatch, capsys):
    dist = {"debug": 0.5, "fix_input": 0.5, "fix_call": 0.0}
    assert invoke(monkeypatch, lambda: FixedBackend(dist), threshold="0.4") == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "unsure" in context
    assert "'debug'" not in context and "'fix_input'" not in context


def test_loud_mode_delivers_classifier_input_and_verdict(monkeypatch, capsys):
    assert invoke(monkeypatch, EncoderContract) == 0
    out = capsys.readouterr().out
    context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("[janus saw]")
    assert "state\ntool: bash" in context
    assert "\n[verdict]\n[janus]" in context


@pytest.mark.parametrize("error", [1, False, {"content": "boom"}, ["boom"]])
def test_json_error_variants_do_not_crash_the_hook(monkeypatch, capsys, error):
    assert invoke(monkeypatch, FixedBackend, payload={**PAYLOAD, "error": error}) == 0


def test_internal_harness_tools_never_get_classified(monkeypatch, capsys):
    for tool in SKIP_TOOLS:
        payload = {**PAYLOAD, "tool_name": tool}
        assert invoke(monkeypatch, FixedBackend, payload=payload) == 0
        assert capsys.readouterr().out == ""


def test_below_threshold_escalates_to_frontier_second_opinion(monkeypatch,
                                                              capsys):
    """Unsure verdicts must carry a frontier second opinion, not stop."""
    dist = {"debug": 0.55, "fix_input": 0.44, "fix_call": 0.01}
    monkeypatch.setenv("JANUS_ESCALATE", "1")
    monkeypatch.setattr(B, "FrontierBackend", lambda **kw: FixedSecond("fix_input"))
    assert invoke(monkeypatch, lambda: FixedBackend(dist)) == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "unsure" in context
    assert "frontier" in context and "fix_input" in context


class FixedSecond:
    name = "frontier"

    def __init__(self, label):
        self.label = label

    def __call__(self, text):
        return {c: (0.7 if c == self.label else 0.15) for c in ("debug", "fix_input", "fix_call")}
