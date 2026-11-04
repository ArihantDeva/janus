"""Public subprocess boundary of the validation scope loop.

The launcher must stop at the first failing scope. A stub runner stands in for
the real gate so the contract is observable without loading a model.
"""

import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "scripts", "run_scopes.sh")


def stub_runner(tmp_path, failing):
    """Record every scope, fail with 7 for `failing`, succeed otherwise."""
    log = tmp_path / "scopes.log"
    script = tmp_path / "stub.sh"
    script.write_text(
        '#!/usr/bin/env bash\n'
        f'echo "$1" >> "{log}"\n'
        f'[ "$1" = "{failing}" ] && exit 7\n'
        'exit 0\n')
    return script, log


def launch(script, scopes):
    return subprocess.run(["bash", RUNNER, str(script), *scopes], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


def test_failing_scope_stops_the_run_and_fails_the_launcher(tmp_path):
    """An early failure masked by a later pass is the defect this guards."""
    script, log = stub_runner(tmp_path, "first")
    result = launch(script, ["first", "second", "third"])
    assert result.returncode != 0, "launcher reported success after a failed scope"
    assert log.read_text().split() == ["first"], "a later scope ran after a failure"


def test_passing_scopes_run_in_order(tmp_path):
    script, log = stub_runner(tmp_path, "none")
    assert launch(script, ["hook", "encoder"]).returncode == 0
    assert log.read_text().split() == ["hook", "encoder"]


def test_no_scope_argument_defaults_to_the_hook_scope(tmp_path):
    script, log = stub_runner(tmp_path, "none")
    assert launch(script, []).returncode == 0
    assert log.read_text().split() == ["hook"]


def test_a_later_failing_scope_still_fails_the_launcher(tmp_path):
    """The guard is order-independent: last-scope failure is not lost either."""
    script, log = stub_runner(tmp_path, "encoder")
    assert launch(script, ["hook", "encoder"]).returncode != 0
    assert log.read_text().split() == ["hook", "encoder"]


def test_sbatch_launcher_uses_the_fail_fast_loop():
    """Wiring: the cluster launcher must call the guarded loop, not inline it."""
    text = open(os.path.join(ROOT, "scripts", "validate_amarel.sbatch")).read()
    assert "run_scopes.sh" in text
    assert "for scope in" not in text
