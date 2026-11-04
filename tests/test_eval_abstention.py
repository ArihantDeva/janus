import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import eval_abstention as A  # noqa: E402


def d(**kw):
    base = {"debug": 0.1, "fix_input": 0.1, "fix_call": 0.1}
    base.update(kw)
    return base


def test_stage_report_partitions_on_threshold():
    rows = [("debug", d(debug=0.9)), ("fix_input", d(debug=0.99)),
            ("fix_call", d(fix_call=0.4)), ("debug", d(fix_input=0.5))]
    r = A.stage_report(rows, 0.60)
    assert r["n"] == 4 and r["answered"] == 2 and r["abstained"] == 2
    assert r["coverage"] == 0.5


def test_stage_report_accuracies_are_stage_local():
    rows = [("debug", d(debug=0.9)), ("fix_input", d(debug=0.99)),
            ("fix_call", d(fix_call=0.4))]
    r = A.stage_report(rows, 0.60)
    assert r["answered_acc"] == 0.5            # one of two answered rows right
    assert r["forced_guess_acc_on_abstained"] == 1.0
    assert r["overall_forced_acc"] == 2 / 3


def test_sweep_covers_each_threshold_once():
    rows = [("debug", d(debug=0.9)), ("debug", d(fix_input=0.55))]
    s = A.sweep(rows, (0.5, 0.6))
    assert [r["threshold"] for r in s] == [0.5, 0.6]
    assert s[0]["answered"] == 2 and s[1]["answered"] == 1


def test_stage_report_empty_abstention_is_none_not_crash():
    r = A.stage_report([("debug", d(debug=0.9))], 0.60)
    assert r["forced_guess_acc_on_abstained"] is None
    assert r["answered_acc"] == 1.0
