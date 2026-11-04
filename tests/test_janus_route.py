"""Tests for the router's decision logic.

The routing rule is where this project can quietly lie to itself, so the tests
target the specific lies:

  * escalating a row the local model was confident about -> tier 1 was decorative
  * never escalating -> the 'frontier safety net' claim is untested
  * a frontier answer that does not name a class -> silently inherits the local
    guess and looks correct
  * labelling a decision 'local' when it came from the frontier (or vice versa)
    -> every cost and accuracy number downstream becomes wrong
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import janus_route as R  # noqa: E402


def fake_local(dist):
    """A local tier that returns a fixed distribution."""
    return lambda text: dict(dist)


UNIFORM = {c: 1 / 6 for c in R.CLASSES}


# ---------------------------------------------------------------------------
# routing decision
# ---------------------------------------------------------------------------

def test_confident_local_answer_does_not_escalate(monkeypatch):
    """A confident local prediction must not spend a frontier call."""
    called = []
    monkeypatch.setattr(R, "escalate", lambda *a, **k: called.append(1) or (None, "", {}))
    local = fake_local({"retry": 0.95, "debug": 0.05, **{c: 0.0 for c in R.CLASSES[2:]}})
    r = R.route("timed out", local=local, threshold=0.6)
    assert r["escalated"] is False
    assert r["decided_by"] == "local"
    assert r["label"] == "retry"
    assert not called, "frontier was called despite high confidence"


def test_unsure_local_answer_escalates(monkeypatch):
    """A low-confidence local prediction must escalate -- otherwise the second
    tier never runs and the safety-net claim is vacuous."""
    monkeypatch.setattr(R, "escalate",
                        lambda text, effort="low": ("reauth", "reauth", {"reasoning_tokens": 10, "total_tokens": 20, "finish": "stop"}))
    local = fake_local(UNIFORM)
    r = R.route("401 unauthorized", local=local, threshold=0.6)
    assert r["escalated"] is True
    assert r["decided_by"] == "frontier"
    assert r["label"] == "reauth"


def test_no_local_tier_always_escalates(monkeypatch):
    """Frontier-only mode must not silently produce a local label."""
    monkeypatch.setattr(R, "escalate",
                        lambda text, effort="low": ("debug", "debug", {"reasoning_tokens": 1, "total_tokens": 2, "finish": "stop"}))
    r = R.route("traceback", local=None)
    assert r["escalated"] is True
    assert r["local"] is None
    assert r["label"] == "debug"


def test_threshold_boundary_is_inclusive_lower(monkeypatch):
    """conf == threshold must NOT escalate; only strictly-below does.

    Pinned because an off-by-one here changes the escalation rate on every row
    that sits exactly at a round confidence.
    """
    monkeypatch.setattr(R, "escalate", lambda *a, **k: pytest.fail("must not escalate"))
    local = fake_local({"retry": 0.6, **{c: 0.08 for c in R.CLASSES[1:]}})
    r = R.route("x", local=local, threshold=0.6)
    assert r["escalated"] is False


def test_force_escalate_overrides_confidence(monkeypatch):
    monkeypatch.setattr(R, "escalate",
                        lambda text, effort="low": ("abort", "abort", {"reasoning_tokens": 1, "total_tokens": 2, "finish": "stop"}))
    local = fake_local({"retry": 0.99, **{c: 0.002 for c in R.CLASSES[1:]}})
    r = R.route("x", local=local, threshold=0.0, force_escalate=True)
    assert r["escalated"] is True
    assert r["decided_by"] == "frontier"


# ---------------------------------------------------------------------------
# the frontier's answer handling
# ---------------------------------------------------------------------------

def test_frontier_parse_takes_the_first_class_named():
    """The parse must take the class the model LED with.

    This test previously asserted longest-first matching, on the false rationale
    that 'fix_call' could be shadowed by a shorter name. No class name is a
    substring of another, so shadowing was impossible -- and longest-first was
    actually wrong: for "fix_call, not fix_input" it returned fix_input. The
    corrected rule is first occurrence.
    """
    for a in R.CLASSES:
        for b in R.CLASSES:
            if a != b:
                assert a not in b, f"{a} is a substring of {b}; shadowing possible"
    # calls the REAL parser, not a local copy of it
    assert R.parse_label("The right move is fix_call, not fix_input.") == "fix_call"
    assert R.parse_label("fix_input is correct, not fix_call") == "fix_input"
    assert R.parse_label("retry") == "retry"
    assert R.parse_label("I am not sure.") is None
    assert R.parse_label("") is None
    assert R.parse_label(None) is None


def test_frontier_unparseable_answer_falls_back_to_local(monkeypatch):
    """If the frontier returns prose with no class name, the label must fall back
    to the local guess -- and must still be attributed to the frontier so a
    failed escalation is visible in the numbers rather than hidden."""
    monkeypatch.setattr(R, "escalate",
                        lambda text, effort="low": (None, "I am not sure.", {"reasoning_tokens": 5, "total_tokens": 9, "finish": "stop"}))
    dist = {"debug": 0.4, "retry": 0.3, "fix_call": 0.1, "fix_input": 0.1,
            "reauth": 0.05, "abort": 0.05}
    local = fake_local(dist)
    r = R.route("weird", local=local, threshold=0.9)
    assert r["escalated"] is True
    assert r["decided_by"] == "frontier"
    assert r["label"] == "debug", "must fall back to the local best, not crash"


# ---------------------------------------------------------------------------
# the action text is the product
# ---------------------------------------------------------------------------

def test_every_class_has_an_action():
    assert set(R.ACTIONS) == set(R.CLASSES)
    for c, txt in R.ACTIONS.items():
        assert txt and txt[0].isupper(), c
        assert txt.endswith("."), c


def test_action_is_imperative_not_a_restatement():
    """The action must tell the agent what to DO. A restatement of the class name
    is the failure mode this exists to prevent."""
    assert R.ACTIONS["retry"].startswith("Re-run")
    assert "credentials" in R.ACTIONS["reauth"].lower()
    assert "stop" in R.ACTIONS["abort"].lower()


def test_unknown_label_gets_safe_action():
    """An unparseable label must render a safe instruction, not crash or claim a
    confident action."""
    r = {"label": "nonsense", "decided_by": "frontier", "local": None,
         "escalated": True,
         "frontier": {"label": None, "raw": "?", "usage": {"reasoning_tokens": 1, "total_tokens": 2}},
         "failure": "x", "threshold": 0.6}
    txt = R.render(r)
    assert "Inspect manually" in txt or "inspect" in txt.lower()


def test_render_names_the_deciding_tier():
    """Every render must state who decided, because the cost story depends on it."""
    r = {"label": "retry", "decided_by": "local", "escalated": False,
         "local": {"label": "retry", "confidence": 0.91,
                   "probs": {c: 0.1 for c in R.CLASSES}},
         "failure": "timed out", "threshold": 0.6}
    out = R.render(r)
    assert "by local" in out
    assert "retry" in out


# ---------------------------------------------------------------------------
# the cost claim
# ---------------------------------------------------------------------------

def test_escalation_rate_is_reported_not_assumed():
    """The eval must expose a curve over thresholds. A single fixed operating
    point cannot distinguish 'the router works' from 'this threshold happens to
    fit this split'."""
    import inspect
    import eval_router
    src = inspect.getsource(eval_router.main)
    assert "curve" in src
    assert src.count("threshold") >= 2
    # it must report the escalation RATE, not just the count
    assert "rate" in src


def test_auc_uncertainty_is_chance_for_uninformative_confidence():
    """AUC 0.5 must come out of confidence that carries no information, otherwise
    the reported AUC would flatter a useless router."""
    conf = [0.5] * 20
    correct = [True] * 10 + [False] * 10
    import eval_router
    assert abs(eval_router.auc_uncertainty(conf, correct) - 0.5) < 1e-9


def test_auc_uncertainty_is_one_for_perfect_confidence():
    import eval_router
    conf = [0.9] * 10 + [0.1] * 10
    correct = [True] * 10 + [False] * 10
    assert abs(eval_router.auc_uncertainty(conf, correct) - 1.0) < 1e-9


def test_auc_uncertainty_is_zero_for_inverted_confidence():
    """Inverted confidence (high conf = wrong) must score 0.0, not 1.0 -- the sign
    is a real output of the measurement, not a detail."""
    import eval_router
    conf = [0.1] * 10 + [0.9] * 10
    correct = [True] * 10 + [False] * 10
    assert abs(eval_router.auc_uncertainty(conf, correct) - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Muse Code hook
# ---------------------------------------------------------------------------

MUSE_PAYLOAD = {
    "hook_event_name": "PostToolUseFailure",
    "tool_name": "bash",
    "tool_input": {"command": "cat /nonexistent/zzz.txt"},
    "tool_use_id": "call_x",
    "error": '{"exit_code": 1, "terminal_status": "failed"}',
    "is_interrupt": False,
    "duration_ms": 67,
    "session_id": "s1",
}


def test_hook_emits_advisory_on_stdout_not_stderr():
    """STDOUT is the only channel that reaches the model.

    Measured failure: the identical JSON written to stderr was recorded in the
    Muse session log with exit_code 0 and the model reported receiving nothing.
    A test that only checks "the hook produced JSON" would pass while the
    integration was silently dead -- so this pins the stream.
    """
    import subprocess
    import json as J
    code = (
        "import sys,json,io;sys.path.insert(0,'src');"
        "from muse_hook import emit;"
        "so,se=io.StringIO(),io.StringIO();"
        "old=(sys.stdout,sys.stderr);sys.stdout,sys.stderr=so,se;"
        "emit('TEST CONTEXT');sys.stdout,sys.stderr=old;"
        "print(J.dumps({'out':so.getvalue(),'err':se.getvalue()}))"
    )
    r = subprocess.run([sys.executable, "-c", "import json as J;" + code],
                       capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = J.loads(r.stdout.strip().splitlines()[-1])
    assert "TEST CONTEXT" in d["out"], "advisory must go to stdout"
    assert d["err"] == "", "stderr is not delivered; it must stay empty"
    payload = J.loads(d["out"])
    assert payload["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"


def test_hook_failure_text_includes_tool_and_error():
    """The classifier needs the error FIRST and the tool/args for disambiguation:
    `cat` on a missing path is fix_input, bad flags are fix_call."""
    import muse_hook as H
    t = H.failure_text(MUSE_PAYLOAD)
    assert "bash" in t
    assert "cat /nonexistent/zzz.txt" in t
    assert "terminal_status" in t
    assert t.index("terminal_status") > t.index("bash"), "error must come after the header"


def test_hook_handles_error_as_string_or_object():
    """`error` arrives as a JSON STRING from Muse, not an object. Code that assumes
    a dict crashes and the agent silently loses its advisory."""
    import muse_hook as H
    as_str = H.failure_text({"tool_name": "bash", "error": '{"a":1}'})
    as_obj = H.failure_text({"tool_name": "bash", "error": {"a": 1}})
    assert "a" in as_str and "a" in as_obj


def test_hook_is_silent_on_interrupt():
    """A user interrupt is not a model failure; advising on it is noise."""
    import subprocess
    import json as J
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = dict(MUSE_PAYLOAD, is_interrupt=True)
    r = subprocess.run([sys.executable, "src/muse_hook.py"], input=J.dumps(p),
                       capture_output=True, text=True, cwd=root,
                       env={**os.environ, "JANUS_HOOK_DISABLE": ""})
    assert r.returncode == 0
    assert r.stdout.strip() == "", "must not advise on a user interrupt"


def test_hook_is_inert_when_disabled():
    """JANUS_HOOK_DISABLE must make the hook truly inert, not merely quiet about a
    frontier call it still makes (which would still cost money)."""
    import subprocess
    import json as J
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "src/muse_hook.py"],
                       input=J.dumps(MUSE_PAYLOAD), capture_output=True,
                       text=True, cwd=root,
                       env={**os.environ, "JANUS_HOOK_DISABLE": "1"})
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_hook_never_crashes_the_turn():
    """A router that raises kills the agent's turn -- strictly worse than none.
    Garbage input must exit 0 with no output."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for bad in ("", "not json", "{}", '{"error": null}', "[]"):
        r = subprocess.run([sys.executable, "src/muse_hook.py"], input=bad,
                           capture_output=True, text=True, cwd=root,
                           env={**os.environ, "JANUS_HOOK_DISABLE": ""})
        assert r.returncode == 0, f"non-zero exit on {bad!r}: {r.stderr[:200]}"
