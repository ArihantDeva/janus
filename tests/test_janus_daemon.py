"""Warm daemon: model built once, client disconnects never kill the loop,
and a daemon miss exits 3 so the caller can fall back to the cold path."""

import io
import json
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import daemon_hook as DH
import janus_daemon as D

PAYLOAD = {"tool_name": "bash", "tool_input": {"command": "pytest -q"},
           "error": "E AssertionError: 1 != 2", "is_interrupt": False}


class CountingBackend:
    name = "janus"
    uses_encoder_input = True

    def __call__(self, text):
        return {"debug": 0.9, "fix_input": 0.07, "fix_call": 0.03}


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    # macOS unix sockets cap at ~104 chars; pytest tmp paths exceed that.
    path = f"/tmp/jt{os.getpid() % 99999}-{id(object()) % 9999}/janus-daemon/hook.sock"
    monkeypatch.setattr(D, "sock_path", lambda: path)
    monkeypatch.delenv("JANUS_HOOK_DISABLE", raising=False)
    calls = []

    def factory():
        calls.append(1)
        return CountingBackend()

    thread = threading.Thread(
        target=D.serve, kwargs={"backend_factory": factory}, daemon=True)
    thread.start()
    ready = D.sock_path() + ".ready"
    for _ in range(200):
        if os.path.exists(ready):
            break
        time.sleep(0.05)
    else:
        pytest.fail("daemon never became ready")
    return calls


def verdict(reply):
    assert reply["ok"], reply
    out = json.loads(reply["stdout"])
    return out["hookSpecificOutput"]["additionalContext"]


def test_two_queries_construct_the_backend_once(daemon):
    for _ in range(2):
        context = verdict(D.query(PAYLOAD, timeout=10))
        assert "[janus] janus read this as debug" in context
    assert len(daemon) == 1, "warm daemon must not rebuild the model per request"


def test_client_disconnects_do_not_kill_the_daemon(daemon):
    for _ in range(3):
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(D.sock_path())
        conn.close()
    assert "[janus] janus read this as debug" in verdict(D.query(PAYLOAD, timeout=10))


def test_unanswered_query_exits_3_for_cold_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        D, "sock_path",
        lambda: f"/tmp/jd{os.getpid() % 99999}/janus-daemon/hook.sock")
    monkeypatch.setenv("JANUS_DAEMON_TIMEOUT", "0.2")
    path = D.sock_path()
    os.makedirs(os.path.dirname(path))
    deaf = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deaf.bind(path)
    deaf.listen(1)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(PAYLOAD)))
    with pytest.raises(SystemExit) as caught:
        D.main()
    assert caught.value.code == 3
    deaf.close()


def test_hook_entry_falls_back_to_direct_on_daemon_miss(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(PAYLOAD)))
    monkeypatch.setattr(DH.D, "query_text",
                        lambda data: (_ for _ in ()).throw(
                            FileNotFoundError("no daemon")))
    monkeypatch.setattr(DH, "cold", lambda data, timeout=280: 7)
    assert DH.main() == 7


def test_hook_entry_daemon_failure_never_breaks_the_turn(monkeypatch, capsys):
    def boom():
        return {"ok": False, "error": "RuntimeError"}

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(PAYLOAD)))
    monkeypatch.setattr(DH.D, "query_text", lambda data: boom())
    monkeypatch.setattr(DH, "cold", lambda data, timeout=280: 0)
    assert DH.main() == 0
