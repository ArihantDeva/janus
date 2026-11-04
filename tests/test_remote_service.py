"""Warm service protocol: framing, isolation, and a bounded request count.

A stub backend stands in for the 1B model; the socket and framing are real.
"""

import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import remote_client as C
import remote_service as S

DIST = {"debug": 0.7, "fix_input": 0.2, "fix_call": 0.1}


@pytest.fixture
def rundir(tmp_path):
    """AF_UNIX paths cap at 104 chars; the macOS sandbox tmp already eats
    ~70. Serve from a short dir so the suite tests framing, not limits."""
    short = "/tmp/janus-rs-%s-%s" % (os.getpid(), tmp_path.name[-8:])
    os.makedirs(short, exist_ok=True)
    yield short
    import shutil
    shutil.rmtree(short, ignore_errors=True)


class Fake:
    name = "janus"

    def __init__(self):
        self.texts = []

    def __call__(self, text):
        self.texts.append(text)
        return DIST


def serve_in_background(backend, job, run_dir, rounds):
    thread = threading.Thread(target=S.serve, args=(backend, job, run_dir, rounds),
                              daemon=True)
    thread.start()
    for _ in range(200):
        if os.path.exists(S.ready_path(job, run_dir)):
            return thread
        thread.join(0.05)
    raise AssertionError("service never signalled readiness")


def test_one_request_gets_the_distribution_and_keeps_text_out_of_the_socketname(
        rundir):
    job, backend = "9001", Fake()
    serve_in_background(backend, job, rundir, 1)
    reply = C.ask(S.socket_path(job, rundir), "state\nboom", 10)
    assert reply["distribution"] == DIST
    assert backend.texts == ["state\nboom"]


def test_socket_is_not_world_reachable(rundir):
    job = "9002"
    serve_in_background(Fake(), job, rundir, 1)
    mode = os.stat(S.socket_path(job, rundir)).st_mode & 0o777
    assert mode == 0o600


def test_empty_or_missing_text_is_refused(rundir):
    job, backend = "9003", Fake()
    serve_in_background(backend, job, rundir, 3)
    path = S.socket_path(job, rundir)
    assert "error" in C.ask(path, "   ", 10)
    assert backend.texts == []


def test_service_removes_its_socket_when_the_allocation_ends(rundir):
    job = "9004"
    thread = serve_in_background(Fake(), job, rundir, 1)
    C.ask(S.socket_path(job, rundir), "state", 10)
    thread.join(10)
    assert not os.path.exists(S.socket_path(job, rundir))
    assert not os.path.exists(S.ready_path(job, rundir))


def test_oversized_frame_is_rejected_not_buffered(monkeypatch):
    """A peer cannot make the service allocate an arbitrary buffer."""
    import socket

    monkeypatch.setattr(S, "MAX_REQUEST_BYTES", 8)
    left, right = socket.socketpair()
    with pytest.raises(ValueError):
        left.sendall((9).to_bytes(4, "big") + b"x" * 9)
        S.recv_message(right)
    left.close()
    right.close()


def test_client_fails_loudly_when_no_allocation_is_named(monkeypatch):
    monkeypatch.delenv("JANUS_REMOTE_JOB", raising=False)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("state"))
    assert C.main() == 2


def test_describe_reports_the_served_checkpoint_and_temperature(monkeypatch, rundir):
    """Calibrated serving must be observable, not assumed."""
    job = "9005"
    monkeypatch.setenv("JANUS_ENCODER", "/scratch/example/model")
    monkeypatch.setenv("JANUS_ENCODER_HEAD", "/scratch/example/model/head.pt")

    class WithT(Fake):
        temperature = 0.5276694558560848

    serve_in_background(WithT(), job, rundir, 1)
    import socket

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(S.socket_path(job, rundir))
    S.send_message(conn, {"describe": True})
    reported = S.recv_message(conn)
    conn.close()
    assert reported["checkpoint"] == "/scratch/example/model"
    assert reported["temperature"] == pytest.approx(0.5276694558560848)
    assert reported["calibrated"] is True
