"""Contract tests for the one-command pipeline.

Criterion: the whole thing must be reproducible from committed scripts. That is
only true if the scripts exist, are wired to each other, and the stages they call
are idempotent. These tests check the wiring, not the science (which the other
suites cover).
"""

import os
import stat
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_pipeline_stage_exists():
    for rel in ("src/build_distill.py", "src/label_janus.py", "src/train_distill.py",
                "src/eval_distill.py", "src/rlcd.py", "src/janus_interface.py",
                "src/encoder_llama.py", "scripts/pipeline.sh",
                "scripts/train_distill.sbatch"):
        assert os.path.exists(os.path.join(ROOT, rel)), f"missing {rel}"


def test_pipeline_script_is_executable_and_syntactically_valid():
    """A pipeline that fails on line 1 is not a pipeline. `bash -n` parses without
    running, so a syntax error is caught without spending a GPU hour."""
    p = os.path.join(ROOT, "scripts", "pipeline.sh")
    assert os.stat(p).st_mode & stat.S_IXUSR, "pipeline.sh is not executable"
    r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sbatch_asks_for_enough_wall_clock():
    """Measured: one run needs ~5.7h (3 cycles of 8213 steps at 1.5s). Asking for 6h
    is a coin flip that discards the whole run; the partition allows 3 days."""
    txt = open(os.path.join(ROOT, "scripts", "train_distill.sbatch")).read()
    hours = None
    for line in txt.splitlines():
        if line.startswith("#SBATCH --time="):
            h, m, s = (line.split("=")[1].strip().split(":"))
            hours = int(h) + int(m) / 60
    assert hours is not None, "no wall clock requested"
    assert hours >= 18, (f"only {hours}h requested. On sm70 hardware the model runs "
                         f"fp32 (fp16 was numerically unreliable), which is ~3x slower "
                         f"per step, so the wall clock must cover that path too")


def test_sbatch_does_not_pin_hardware_to_a_node_class():
    """A constraint on ampere|adalovelace narrowed capacity until the job sat in
    Priority for 20+ minutes. The V100 slowness was a dtype bug, fixed in code."""
    txt = open(os.path.join(ROOT, "scripts", "train_distill.sbatch")).read()
    assert "--constraint=" not in txt


def test_sbatch_covers_the_seeds_and_the_controls():
    """Three seeds for the spread, plus the ablations. A missing variant would be a
    silently absent control.

    The array bounds are passed on the sbatch command line rather than in the file,
    so only the variant table is checkable from the file itself.
    """
    txt = open(os.path.join(ROOT, "scripts", "train_distill.sbatch")).read()
    for v in ("--seed 0", "--seed 1", "--seed 2", "--freeze-encoder",
              "--rlcd-strength 0", "--temp 1.0", "--min-conf 0.6", "--maxlen 8192"):
        assert v in txt, f"variant flag missing from sbatch: {v}"
    # the three seed variants must be distinct tasks, not three copies of seed 0
    assert txt.count("--seed 0") >= 1 and "--seed 1" in txt and "--seed 2" in txt


def test_makefile_exposes_both_entry_points():
    txt = open(os.path.join(ROOT, "Makefile")).read()
    assert "distill:" in txt and "eval-distill:" in txt


def test_labeller_is_resumable_by_construction():
    """Idempotence is what makes the pipeline re-runnable. The labeller must skip
    rows it has already labelled, keyed on the full state."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import label_janus as L
    a = {"tool": "bash", "arguments": {"command": "x"}, "error": "boom"}
    b = {"tool": "bash", "arguments": {"command": "y"}, "error": "boom"}
    assert L.state_key(a) == L.state_key(dict(reversed(list(a.items()))))
    assert L.state_key(a) != L.state_key(b), "arguments must be part of the key"


def test_refuses_to_train_on_cpu_inside_a_slurm_allocation(monkeypatch, tmp_path):
    """Measured: a job landed on gpu017, printed `CUDA initialization: CUDA unknown
    error`, silently fell back to CPU, and spent 485s on ONE step. On a SLURM
    allocation that must be a loud failure, not a 100x-slower success."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "train_distill.py"),
         "--out", str(tmp_path), "--epochs1", "0", "--epochs2", "0"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    assert "Refusing to train on CPU" in (r.stdout + r.stderr)
