"""Tests for seed aggregation.

The criterion is a mean+spread across three seeds. The failure this guards against is
averaging in a run that diverged, which makes the spread look tighter than reality and
is exactly what happened when fp16 destroyed one seed while the others trained.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import aggregate as A


def _res(seed, acc, ece=0.03, complete=True, nonfinite=0, floor=0.55, kw=0.31):
    return {"seed": seed, "complete": complete, "tool_only_floor": floor,
            "keyword_null": kw,
            "results": {"distill": {"acc": acc, "ece": ece,
                                    "nonfinite_steps": nonfinite}}}


def test_usable_requires_complete():
    ok, why = A.usable(_res(0, 0.8, complete=False))
    assert not ok and "incomplete" in why[0]


def test_usable_excludes_a_seed_that_went_non_finite():
    """The measured case: seed 2 read `loss nan` from step 5000 while 0/1/3 trained.
    Averaging it in would report a spread the model does not have."""
    ok, why = A.usable(_res(2, 0.0, nonfinite=3))
    assert not ok
    assert "non-finite" in why[0]


def test_usable_accepts_a_clean_run():
    ok, why = A.usable(_res(0, 0.8))
    assert ok and why == []


def test_summarize_reports_mean_stdev_and_spread():
    s = A.summarize([0.70, 0.80, 0.90])
    assert s["mean"] == pytest.approx(0.80)
    assert s["n"] == 3
    assert s["spread"] == pytest.approx(0.20)
    assert s["stdev"] > 0


def test_summarize_of_one_seed_reports_zero_spread_not_an_error():
    """A single seed is a legitimate intermediate state; reporting stdev 0 with n=1 is
    honest as long as n is shown. Crashing would block a partial check."""
    s = A.summarize([0.8])
    assert s["n"] == 1 and s["stdev"] == 0.0


def test_summarize_of_nothing_is_none_not_zero():
    """Returning 0.0 for 'no data' would read as a measuring a perfect score."""
    assert A.summarize([]) is None


def test_aggregate_end_to_end_excludes_and_reports(tmp_path):
    for seed, acc, nf in [(0, 0.70, 0), (1, 0.72, 0), (2, 0.0, 5)]:
        json.dump(_res(seed, acc, nonfinite=nf),
                  open(tmp_path / f"result_seed{seed}.json", "w"))
    out = tmp_path / "agg.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "aggregate.py"),
                        "--dir", str(tmp_path), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "EXCLUDED result_seed2.json" in r.stdout
    agg = json.load(open(out))
    assert agg["n_usable"] == 2 and agg["n_excluded"] == 1
    # the diverged seed must not drag the mean down to 0.47
    assert agg["aggregate"]["distill"]["acc"]["mean"] == pytest.approx(0.71)


def test_aggregate_refuses_when_nothing_is_usable(tmp_path):
    json.dump(_res(0, 0.7, complete=False), open(tmp_path / "result_seed0.json", "w"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "aggregate.py"),
                        "--dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0
    assert "refusing" in (r.stdout + r.stderr).lower()
