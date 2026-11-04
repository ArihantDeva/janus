"""Hook behavior when the durable remote route fails: fail open, never advise."""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import backends as B
import muse_hook as H
import remote_slurm as R

PAYLOAD = {"tool_name": "bash", "tool_input": {"command": "cat /nope"},
           "error": "cat: /nope: No such file or directory"}


@pytest.fixture(autouse=True)
def enabled_hook(monkeypatch):
    monkeypatch.delenv("JANUS_HOOK_DISABLE", raising=False)


def invoke(monkeypatch, factory, payload=PAYLOAD):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(B, "get_backend", factory)
    monkeypatch.setenv("JANUS_THRESHOLD", "0.60")
    return H.main()


def backend_that_raises(error):
    def factory():
        raise error
    return factory


@pytest.mark.parametrize("error", [R.Unavailable("no remote"), R.Timeout("slow"),
                                   OSError("ssh"), ValueError("bad")])
def test_remote_failure_names_no_class_and_does_not_break_the_turn(monkeypatch,
                                                                   capsys, error):
    assert invoke(monkeypatch, backend_that_raises(error)) == 0
    out = capsys.readouterr()
    assert out.out == "", "a failure must not fabricate an advisory"
    for name in ("'debug'", "'fix_input'", "'fix_call'"):
        assert name not in out.out


def test_failure_diagnostics_are_sanitized_but_visible(monkeypatch, capsys):
    invoke(monkeypatch, backend_that_raises(R.Unavailable("ssh ad2416@host secret")))
    out = capsys.readouterr()
    assert "Unavailable" in out.err
    assert "secret" not in out.err and "ad2416" not in out.err


def test_disabled_hook_never_constructs_the_remote_backend(monkeypatch, capsys):
    def forbidden():
        pytest.fail("disabled hook must not construct a backend")

    monkeypatch.setenv("JANUS_HOOK_DISABLE", "1")
    assert invoke(monkeypatch, forbidden) == 0
    assert capsys.readouterr().out == ""


def test_remote_backend_receives_the_canonical_encoder_input(monkeypatch, capsys):
    """The remote must be asked exactly what training used: state/question/choices."""
    seen = {}

    class Encoder(R.RemoteEncoder):
        name = "janus"

        def __call__(self, text):
            seen["text"] = text
            return {"debug": 0.1, "fix_input": 0.8, "fix_call": 0.1}

    assert invoke(monkeypatch, Encoder) == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert seen["text"].startswith("state\n")
    assert "\n\nchoices\n" in seen["text"]
    assert "answer:" not in seen["text"]
    assert "fix_input" in context and "janus" in context
