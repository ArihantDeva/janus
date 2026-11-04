"""Cluster launcher: ownership, staleness and deadline behavior of the SSH route.

`LocalLauncher` is replaced by a stub `run` callable, so no Slurm job is created.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import remote_slurm as R

DIST = {"debug": 0.7, "fix_input": 0.2, "fix_call": 0.1}


class Calls:
    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def __call__(self, argv, timeout, stdin=None):
        self.seen.append((argv, timeout, stdin))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def launcher(answers, **kw):
    calls = Calls(answers)
    return R.SlurmLauncher(run=calls, **kw), calls


def test_start_returns_the_parsed_owned_job_id():
    lch = R.SlurmLauncher(run=Calls(["4321\n", ""]), timeout=5)
    assert lch.start() == "4321"


def test_start_waits_for_readiness_before_returning():
    """A submitted job that never opened its socket is not a usable allocation."""
    lch = R.SlurmLauncher(run=Calls(["4321\n", OSError("not ready")]), timeout=5)
    with pytest.raises(R.Unavailable):
        lch.start()


def test_start_rejects_non_numeric_job_output():
    """An unparsed sbatch line must not become a job id."""
    lch = R.SlurmLauncher(run=Calls(["sbatch: error: invalid partition\n"]),
                          timeout=5)
    with pytest.raises(R.Unavailable):
        lch.start()


def test_leaving_node_is_the_transport_not_the_login_node():
    lch, calls = launcher(["4321\n", "", json.dumps({"distribution": DIST})])
    lch.start()
    assert lch.request("4321", "state", 5) == pytest.approx(DIST)
    script = " ".join(str(part) for part in calls.seen[-1][0])
    assert "srun" in script and "--jobid=4321" in script
    assert "ssh" not in script
    assert calls.seen[-1][2] == "state", "the encoder input travels on stdin"


def test_request_returns_the_distribution_not_the_envelope():
    lch, _ = launcher(["4321\n", "", json.dumps({"distribution": DIST})])
    lch.start()
    assert lch.request("4321", "state", 5) == pytest.approx(DIST)


def test_clean_stdout_is_required_for_parsing():
    lch, _ = launcher(["4321\n", "",
                       "loading weights\n" + json.dumps({"distribution": DIST})])
    lch.start()
    with pytest.raises(R.Unavailable):
        lch.request("4321", "state", 5)


def test_released_allocation_is_reported_as_stale():
    """srun against a job that expired fails; that is stale, not a transient error."""
    lch, _ = launcher(["4321\n", "", R.Stale("srun: error: invalid job id")])
    lch.start()
    with pytest.raises(R.Stale):
        lch.request("4321", "state", 5)


def test_timeout_is_not_mistaken_for_staleness():
    lch, _ = launcher(["4321\n", "", TimeoutError("deadline")])
    lch.start()
    with pytest.raises(R.Timeout):
        lch.request("4321", "state", 5)


def test_ssh_failure_is_unavailable_not_stale():
    lch, _ = launcher(["4321\n", "", OSError("connection refused")])
    lch.start()
    with pytest.raises(R.Unavailable):
        lch.request("4321", "state", 5)


def test_describe_reports_the_deployed_script_not_a_scratch_probe():
    """Verification must use a path that ships with the deployment."""
    lch, calls = launcher(["4321\n", "", json.dumps({"serving": True})])
    lch.start()
    assert lch.describe("4321") == {"serving": True}
    script = " ".join(str(part) for part in calls.seen[-1][0])
    assert "scripts/remote_describe.sh" in script and "probe" not in script
