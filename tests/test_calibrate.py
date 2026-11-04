"""Tests for post-hoc temperature scaling.

The criterion is ECE <= 0.05, so the calibration code has to be verified on cases where
the right answer is known, not on a model. Specifically: it must not change accuracy,
it must reduce ECE on a deliberately overconfident predictor, and it must refuse to
fit on rows it was told to stay away from.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import calibrate as C


def _overconfident(n=2000, seed=0):
    """A predictor that is often right but always says 0.99.

    This is the shape distillation produces: trained to KL-match a teacher whose median
    confidence is 0.93, so it learns to be peaked regardless of whether it is correct.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    correct = rng.random(n) < 0.75
    logits = np.zeros((n, 3))
    logits[np.arange(n), y] = 6.0                      # big margin on the right class
    wrong = ~correct
    logits[wrong] = 0.0
    logits[wrong, (y[wrong] + 1) % 3] = 6.0            # confidently wrong
    return logits, y


# ---------------------------------------------------------------------------
# ECE itself
# ---------------------------------------------------------------------------

def test_ece_is_zero_for_a_perfectly_calibrated_confident_predictor():
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert C.expected_calibration_error(P, np.array([0, 1])) == pytest.approx(0.0)


def test_ece_at_high_confidence_rewards_corresponding_accuracy():
    """What ECE actually measures, and a mistake I made writing this test.

    I first asserted that a 0.99-confident predictor always scores WORSE than a
    0.72-confident one on 4 correct rows. It does not: ECE is
    |confidence - accuracy|, so at 4/4 accuracy the 0.99 predictor scores 0.010 and the
    0.72 predictor scores 0.280. High confidence is only punished when accuracy does not
    match it. My first version confused "overconfident" with "confident" -- and that
    distinction is the entire content of a calibration claim.
    """
    y_all_correct = np.array([0, 0, 1, 1])
    confident = np.array([[0.99, 0.005, 0.005], [0.99, 0.005, 0.005],
                          [0.005, 0.99, 0.005], [0.005, 0.99, 0.005]])
    understated = np.array([[0.72, 0.14, 0.14]] * 2 + [[0.14, 0.72, 0.14]] * 2)
    assert C.expected_calibration_error(confident, y_all_correct) < \
           C.expected_calibration_error(understated, y_all_correct)

    # ...but the same 0.99 confidence on rows that are only HALF right is worse than
    # anything honest: that is overconfidence proper
    y_half = np.array([0, 1, 1, 0])
    assert C.expected_calibration_error(confident, y_half) > \
           C.expected_calibration_error(understated, y_half)


def test_ece_rejects_string_labels():
    """The same defect that reported acc 0.0% in training: string labels make
    `pred == y` always False, which inflates ECE to ~1.0 and looks like a broken
    model rather than a broken call."""
    P = np.array([[0.7, 0.2, 0.1]])
    with pytest.raises(TypeError, match="integer class indices"):
        C.expected_calibration_error(P, np.array(["debug"]))


def test_ece_is_bounded():
    """ECE is an absolute difference of two quantities in [0,1], so it cannot exceed 1.
    A value above 1 means the binning is wrong."""
    logits, y = _overconfident(500)
    P = C.apply_temperature(logits, 0.05)
    assert 0.0 <= C.expected_calibration_error(P, y) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# temperature scaling
# ---------------------------------------------------------------------------

def test_temperature_one_is_a_no_op():
    z = np.array([[2.0, 0.5, -1.0]])
    p1 = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert C.apply_temperature(z, 1.0) == pytest.approx(p1)


def test_temperature_cannot_change_the_argmax():
    """This is the load-bearing property: accuracy is untouched by construction. A
    calibration method that could flip predictions would make the ECE claim and the
    accuracy claim interact, and the reported accuracy would depend on whether
    calibration ran."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 3)) * 3
    base = z.argmax(1)
    for t in (0.1, 0.5, 1.0, 2.0, 10.0):
        assert (C.apply_temperature(z, t).argmax(1) == base).all(), t


def test_fitting_reduces_ece_on_an_overconfident_predictor():
    """The actual claim. Fit on one set, measure on another: fitting and measuring on
    the same rows would report a number the model does not have."""
    z_fit, y_fit = _overconfident(3000, seed=0)
    z_ev, y_ev = _overconfident(3000, seed=1)

    t = C.fit_temperature(z_fit, y_fit)
    before = C.expected_calibration_error(C.apply_temperature(z_ev, 1.0), y_ev)
    after = C.expected_calibration_error(C.apply_temperature(z_ev, t), y_ev)
    print(f"  fitted T={t:.3f}  ECE {before:.4f} -> {after:.4f}")
    assert after < before, (before, after)
    assert t > 1.0, "an overconfident model must be softened, not sharpened"


def test_fitted_temperature_leaves_accuracy_unchanged():
    z, y = _overconfident(1000)
    acc_before = (C.apply_temperature(z, 1.0).argmax(1) == y).mean()
    t = C.fit_temperature(z, y)
    acc_after = (C.apply_temperature(z, t).argmax(1) == y).mean()
    assert acc_before == acc_after


def test_fitting_finds_a_sensible_temperature_for_a_known_case():
    """Sensitivity check: if the fitter returned 1.0 for a wildly overconfident
    predictor, the reduction test above could still pass by luck of the grid."""
    z, y = _overconfident(2000)
    t = C.fit_temperature(z, y)
    assert 1.05 < t < 10.0, f"T={t} is implausible for a 0.99-confidence predictor"


def test_nll_objective_is_available_and_differs_from_ece():
    """Two objectives, reported side by side. If they always agreed there would be no
    reason to expose the choice."""
    z, y = _overconfident(2000)
    t_ece = C.fit_temperature(z, y, objective="ece")
    t_nll = C.fit_temperature(z, y, objective="nll")
    assert t_ece > 1.0 and t_nll > 1.0
    assert t_ece != pytest.approx(t_nll)


def test_invalid_temperature_and_objective_are_rejected():
    z, y = _overconfident(50)
    with pytest.raises(ValueError):
        C.apply_temperature(z, 0.0)
    with pytest.raises(ValueError):
        C.fit_temperature(z, y, objective="brier-ish")


# ---------------------------------------------------------------------------
# the disjointness guard
# ---------------------------------------------------------------------------

def test_assert_disjoint_refuses_shared_groups():
    """Fitting T on eval rows is fitted-on-test: it reports an ECE the deployed model
    does not have. The guard must fire."""
    train = [{"group": "a"}, {"group": "b"}]
    calib = [{"group": "b"}, {"group": "c"}]
    with pytest.raises(AssertionError, match="disjoint"):
        C.assert_disjoint(train, calib)


def test_assert_disjoint_passes_for_clean_splits():
    C.assert_disjoint([{"group": "a"}], [{"group": "b"}], [{"group": "c"}])


def test_calibration_guard_fires_on_the_leak_three_way_split_creates():
    """Reintroduce the real defect: `split_by_group` produces a DISJOINT pair, so the
    natural way to add a calibration set is to slice the training rows again -- which
    reuses training groups and is the leak this guard exists for. (An earlier version of
    this test assumed split_by_group itself leaked, which it does not; the defect is
    the third split, not the second.)"""
    from train_distill import split_by_group
    rows = [{"group": f"g{i//5}", "tool": "bash", "error": "x", "arguments": {},
             "teacher": {"debug": 1.0, "fix_input": 0.0, "fix_call": 0.0},
             "teacher_argmax": "debug"} for i in range(100)]
    tr, ev, _ = split_by_group(rows, frac=0.2, seed=0)
    calib = tr[:20]                       # the careless move: reuse training rows
    with pytest.raises(AssertionError, match="disjoint"):
        C.assert_disjoint(tr, calib, ev)


# ---------------------------------------------------------------------------
# threshold selection
# ---------------------------------------------------------------------------

def _curve(seed=0, n=4000):
    """A predictor whose accuracy genuinely rises with stated confidence.

    My first version drew `conf` and `right` INDEPENDENTLY, so accuracy was flat in
    confidence (measured 0.5525 / 0.5495 / 0.5707 across thresholds) and the tests
    asserting monotonicity correctly failed. A fixture without the property under test
    cannot test it. Here correctness probability is `conf` itself, which is exactly
    the calibrated case, plus a per-row margin that is large when the model is right.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    conf = rng.random(n) * 0.85 + 0.12          # stated confidence, 0.12..0.97
    right = rng.random(n) < conf                # accuracy tracks confidence
    chosen = np.where(right, y, (y + 1) % 3)
    P = np.full((n, 3), ((1.0 - conf) / 2)[:, None])   # broadcast per row
    P[np.arange(n), chosen] = conf
    P = P / P.sum(1, keepdims=True)
    return P, y


def test_threshold_reaches_a_target_accuracy_with_useful_coverage():
    """The product question: how much can be answered at 85% accuracy?

    Note the coverage floor has to be set below the coverage the target implies.
    Measured on this fixture: 85% accuracy is reached at ~14% coverage, so the default
    min_coverage of 0.5 makes the target unreachable and the function correctly returns
    None. My first version of this test left the default in place and read the None as a
    bug in the code; it was a contradiction in the request.
    """
    P, y = _curve()
    tau, cov, acc = C.select_threshold(P, y, target_accuracy=0.85, min_coverage=0.05)
    assert tau is not None, "85% should be reachable on a calibrated-ish predictor"
    assert acc >= 0.85 and cov > 0.0


def test_threshold_is_none_when_the_target_contradicts_the_coverage_floor():
    """An unreachable request must return None rather than the best effort, because a
    returned threshold is treated by the caller as meeting the target."""
    """Returning a threshold that does not meet the target would be worse than
    admitting the target is out of reach."""
    P, y = _curve()
    tau, cov, acc = C.select_threshold(P, y, target_accuracy=0.999, min_coverage=0.5)
    assert tau is None


def test_threshold_respects_the_coverage_floor():
    """A threshold answering 2 rows brilliantly is not an operating point. The floor
    is what stops the search from returning one."""
    P, y = _curve()
    tau, cov, acc = C.select_threshold(P, y, min_coverage=0.5)
    assert cov >= 0.5


def test_higher_threshold_never_lowers_accuracy_on_a_calibrated_predictor():
    """The monotonicity that makes a threshold meaningful at all. If accuracy did not
    rise with confidence, the hook could not use probability as a decision signal."""
    P, y = _curve()
    accs = []
    for tau in (0.4, 0.6, 0.8):
        m = P.max(1) >= tau
        accs.append((P.argmax(1)[m] == y[m]).mean())
    assert accs[0] < accs[1] < accs[2], accs


def test_threshold_is_fitted_on_calib_not_eval():
    """The guard that makes the reported operating point honest: a threshold fitted on
    eval rows would not be the deployed one. Same discipline as temperature."""
    P_cal, y_cal = _curve(seed=0)
    P_ev, y_ev = _curve(seed=99)
    tau, _, _ = C.select_threshold(P_cal, y_cal, target_accuracy=0.8, min_coverage=0.05)
    assert tau is not None
    m = P_ev.max(1) >= tau
    # the threshold transfers and still answers a useful fraction of unseen rows
    assert m.mean() > 0.05, "a threshold that answers almost nothing on new data is useless"
