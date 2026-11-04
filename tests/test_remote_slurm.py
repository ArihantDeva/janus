"""Durable Amarel transport: ownership, staleness, deadlines, fail-open.

The cluster is replaced by a stub launcher, so ownership and staleness behavior
are observable without Slurm. `RemoteEncoder` is the public seam.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import remote_slurm as R

DIST = {"debug": 0.7, "fix_input": 0.2, "fix_call": 0.1}
UNSET = object()


class StubLauncher:
    """Swap the cluster: `start` owns a job, `request` answers for one."""

    def __init__(self, reply=UNSET, start_error=None, request_error=None):
        self.reply = DIST if reply is UNSET else reply
        self.start_error = start_error
        self.request_error = request_error
        self.alive_answer = True
        self.started = 0
        self.requests = []

    def start(self):
        self.started += 1
        if self.start_error:
            raise self.start_error
        return "1234"

    def request(self, job, text, timeout):
        self.requests.append((job, text, timeout))
        if self.request_error:
            raise self.request_error
        return self.reply

    def alive(self, job):
        return self.alive_answer

    def cleanup(self, job):
        self.cleaned = getattr(self, "cleaned", []) + [job]


def client(tmp_path, launcher, **kw):
    return R.RemoteEncoder(launcher=launcher, state_dir=str(tmp_path), **kw)


def recorded(tmp_path, job="999"):
    (tmp_path / "allocation").write_text(job)
    return str(tmp_path)


def test_reachable_happy_path_returns_a_validated_distribution(tmp_path):
    launcher = StubLauncher()
    assert client(tmp_path, launcher)("state\nboom") == pytest.approx(DIST)
    assert launcher.requests[0][1] == "state\nboom"


def test_validated_output_keeps_the_fixed_class_order(tmp_path):
    out = client(tmp_path, StubLauncher())("state\nboom")
    assert list(out) == ["debug", "fix_input", "fix_call"]


def test_warm_requests_reuse_one_owned_allocation(tmp_path):
    launcher = StubLauncher()
    backend = client(tmp_path, launcher)
    backend("one")
    backend("two")
    assert launcher.started == 1
    assert [job for job, _t, _d in launcher.requests] == ["1234", "1234"]


def test_a_recorded_live_allocation_is_reused(tmp_path):
    """A fresh process must find the owned allocation instead of starting one."""
    launcher = StubLauncher()
    assert client(recorded(tmp_path), launcher)("state") == pytest.approx(DIST)
    assert launcher.requests[0][0] == "999"
    assert launcher.started == 0


@pytest.mark.parametrize("error", [R.Unavailable("boom"), OSError("ssh down")])
def test_missing_remote_fails_open_without_advice(tmp_path, error):
    launcher = StubLauncher(request_error=error)
    with pytest.raises(R.Unavailable):
        client(tmp_path, launcher)("state\nboom")


def test_start_failure_is_reported_as_unavailable(tmp_path):
    with pytest.raises(R.Unavailable):
        client(tmp_path, StubLauncher(start_error=OSError("sbatch")))("state")


def test_stale_allocation_is_replaced_by_a_new_owned_job(tmp_path):
    """A released allocation must recover on the next request, not on a timer."""
    launcher = StubLauncher()
    backend = client(recorded(tmp_path), launcher)
    launcher.alive_answer = False
    launcher.request_error = R.Stale("go")

    def answer_later(job, text, timeout):
        launcher.requests.append((job, text, timeout))
        if job != "1234":
            raise R.Stale("go")
        return DIST

    launcher.request = answer_later
    assert backend("state") == pytest.approx(DIST)
    assert launcher.started == 1
    assert launcher.requests[-1][0] == "1234"


def test_stale_allocation_that_keeps_failing_is_bounded(tmp_path):
    """One replacement, then give up: no unbounded restart loop."""
    launcher = StubLauncher(request_error=R.Stale("gone"))
    with pytest.raises(R.Unavailable):
        client(tmp_path, launcher)("state")
    assert launcher.started == 1


def test_timeout_is_bounded_and_surfaces_as_unavailable(tmp_path):
    launcher = StubLauncher(request_error=R.Timeout("slow"))
    with pytest.raises(R.Unavailable):
        client(tmp_path, launcher, timeout=1)("state")
    assert launcher.requests[0][2] == 1


@pytest.mark.parametrize("reply", [None, "not json", {}, {"debug": 1.0},
                                   {"debug": 2.0, "fix_input": -1.0,
                                    "fix_call": 0.0}])
def test_malformed_payload_never_becomes_advice(tmp_path, reply):
    with pytest.raises(R.Unavailable):
        client(tmp_path, StubLauncher(reply=reply))("state")


def test_class_names_come_from_the_encoder_interface(tmp_path):
    import janus_interface as JI

    with pytest.raises(R.Unavailable):
        client(tmp_path, StubLauncher({"ok": 1.0, "error": 0.0, "other": 0.0}))("state")
    assert JI.CLASSES_3 == ("debug", "fix_input", "fix_call")


def test_deadline_is_passed_through_unchanged(tmp_path):
    launcher = StubLauncher()
    client(tmp_path, launcher, timeout=9)("state")
    assert launcher.requests[0][2] == 9


def test_replacing_a_released_allocation_clears_its_leftover_socket(tmp_path):
    """A released allocation's socket is dead weight, not a usable service."""
    launcher = StubLauncher()
    launcher.alive_answer = False
    client(recorded(tmp_path, "999"), launcher)._restart()
    assert launcher.cleaned == ["999"]


def test_encoder_defaults_to_local_so_direct_callers_keep_their_weights(monkeypatch):
    """Only the installed hook opts into the cluster; construction stays local."""
    import backends as B

    monkeypatch.delenv("JANUS_ENCODER_TRANSPORT", raising=False)
    assert B.encoder_transport() == "local"


def test_installed_configuration_carries_no_job_id():
    """Ownership is discovered at request time; nothing durable names a job."""
    from install_muse_hook import command_for

    command = command_for("/home/example/januslab").lower()
    assert "jobid" not in command and "srun" not in command


def test_a_second_hook_event_does_not_start_a_second_allocation(tmp_path):
    """Two concurrent failures must converge on one owned allocation."""
    launcher = StubLauncher()
    launcher.alive_answer = False
    backend = client(recorded(tmp_path, "111"), launcher)
    backend._restart()
    assert launcher.started == 1

    late = client(tmp_path, launcher)  # constructed while 111 was still recorded
    launcher.alive_answer = True
    late._restart()
    assert launcher.started == 1, "the recorded live allocation must be adopted"
    assert late._job == "1234"
