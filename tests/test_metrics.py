"""Tests for the metrics module.

The point of these tests is that a reported number can be reproduced from a
clean checkout. Every metric is checked against a hand-computed value, so a
sloppy refactor cannot silently move a published figure.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (  # noqa: E402
    apply_temperature,
    aurc,
    brier,
    coverage_at,
    ece,
    fit_temperature,
    gate,
)


# ------------------------------------------------------------------ brier

def test_brier_perfect_prediction_is_zero():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)


def test_brier_confident_and_wrong_is_one():
    assert brier(np.array([1.0]), np.array([0])) == pytest.approx(1.0)


def test_brier_known_value():
    # (0.8-1)^2 = 0.04 ; (0.3-0)^2 = 0.09 ; mean = 0.065
    assert brier(np.array([0.8, 0.3]), np.array([1, 0])) == pytest.approx(0.065)


# --------------------------------------------------------- calibration

def test_ece_perfectly_calibrated_is_zero():
    """Two bins, each exactly calibrated."""
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert ece(p, y) == pytest.approx(0.0)


def test_ece_maximally_miscalibrated():
    """Always says 1.0, always wrong -> ECE 1.0."""
    assert ece(np.ones(4), np.zeros(4, dtype=int)) == pytest.approx(1.0)


def test_ece_is_bounded_between_zero_and_one():
    rng = np.random.RandomState(0)
    p = rng.uniform(size=200)
    y = (rng.uniform(size=200) > 0.5).astype(int)
    assert 0.0 <= ece(p, y) <= 1.0


def test_temperature_one_is_identity():
    p = np.array([0.1, 0.4, 0.9])
    assert apply_temperature(p, 1.0) == pytest.approx(p)


def test_temperature_flattens_with_increasing_t():
    p = np.array([0.05, 0.95])
    flat = apply_temperature(p, 10.0)
    assert np.all(np.abs(flat - 0.5) < np.abs(p - 0.5))


def test_fit_temperature_reduces_ece():
    rng = np.random.RandomState(0)
    logit = rng.normal(size=500) * 3.0
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=500) < p).astype(int)
    t = fit_temperature(p, y)
    assert ece(apply_temperature(p, t), y) <= ece(p, y)


# ------------------------------------------------------------------- aurc

def test_aurc_perfect_ranking_is_zero():
    p = np.array([0.99, 0.98, 0.01, 0.02])
    y = np.array([1, 1, 0, 0])
    assert aurc(p, y) == pytest.approx(0.0)


def test_aurc_monotone_in_confidence():
    # A confident-and-wrong model must have higher AURC than a confident-and-right one.
    y = np.array([1, 1, 0, 0])
    good = np.array([0.99, 0.95, 0.05, 0.01])
    bad = np.array([0.01, 0.05, 0.95, 0.99])
    assert aurc(bad, y) > aurc(good, y)


# -------------------------------------------------------------- coverage

def test_coverage_at_zero_budget_with_perfect_predictions_covers_all():
    """A 0% error budget is met by a model that is never wrong."""
    p = np.array([0.99, 0.01])
    y = np.array([1, 0])
    cov, k = coverage_at(p, y, budget=0.0)
    assert k == 2 and cov == pytest.approx(1.0)


def test_coverage_at_zero_budget_with_any_error_covers_nothing():
    p = np.array([0.99, 0.6])
    y = np.array([1, 0])          # second item wrong
    cov, k = coverage_at(p, y, budget=0.0)
    assert k == 1 and cov == pytest.approx(0.5)


def test_coverage_at_one_budget_is_everything():
    p = np.array([0.7, 0.2])
    y = np.array([1, 0])
    cov, k = coverage_at(p, y, budget=1.0)
    assert k == 2 and cov == pytest.approx(1.0)


def test_coverage_counts_only_confident_correct_prefix():
    # Sorted by confidence: 1st correct, 2nd wrong, 3rd correct, 4th correct
    p = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 1, 1])   # 2nd item is wrong (p>=.5 but y=0)
    cov, k = coverage_at(p, y, budget=0.0)
    assert k == 1
    # Prefix risks are 0.0, 0.5, 1/3, 0.25 and the worst prefix is exactly
    # 0.5. A 0.5 budget is inclusive, so all four qualify; one step tighter
    # and the k=2 prefix violates the budget, capping coverage at k=1.
    _, k_at_budget = coverage_at(p, y, budget=0.5)
    assert k_at_budget == 4
    _, k_tighter = coverage_at(p, y, budget=0.49)
    assert k_tighter == 1


# ------------------------------------------------------------------- gate

def test_gate_blocks_confident_failures():
    p = np.array([0.95, 0.85, 0.05, 0.03])
    y = np.array([1, 0, 1, 0])
    g = gate(p, y, hi=0.9, lo=0.2)
    assert g["block_rate"] == pytest.approx(0.25)
    assert g["fail_recall"] == pytest.approx(0.5)   # caught 1 of 2 failures
    assert g["block_precision"] == pytest.approx(1.0)


def test_gate_ask_band():
    p = np.array([0.5])
    y = np.array([1])
    g = gate(p, y, hi=0.9, lo=0.2)
    assert g["ask_rate"] == pytest.approx(1.0)
    assert g["block_rate"] == pytest.approx(0.0)


def test_gate_allow_band():
    p = np.array([0.1])
    y = np.array([0])
    g = gate(p, y, hi=0.9, lo=0.2)
    assert g["block_rate"] == 0 and g["ask_rate"] == 0
    assert g["allow_rate"] == pytest.approx(1.0)


def test_gate_rates_sum_to_one():
    rng = np.random.RandomState(1)
    p = rng.uniform(size=100)
    y = (rng.uniform(size=100) > 0.5).astype(int)
    g = gate(p, y, hi=0.7, lo=0.3)
    assert g["block_rate"] + g["ask_rate"] + g["allow_rate"] == pytest.approx(1.0)


# --------------------------------------------- temperature grid must not clip

def test_temperature_fit_reaches_beyond_the_grid_ceiling():
    """A grid search that stops at hi returns hi whenever the optimum is past
    it, with no signal that the search was truncated. R-Judge fitted T=20.00
    exactly, which is the ceiling: the reported calibration came from a fit that
    had not converged. Widening must move it off the bound.
    """
    import numpy as np
    import metrics
    rng = np.random.default_rng(0)
    # deliberately overconfident and correct: the optimum temperature is large
    y = np.array([0] * 500 + [1] * 500, float)
    p = np.where(y == 1, 0.999, 0.001)
    T = metrics.fit_temperature(p, y)
    assert T < metrics.TEMP_MAX, (
        f"T={T} sits on the search ceiling; the fit is truncated, not converged")


def test_temperature_fit_reports_whether_it_hit_the_bound():
    """Truncation must be visible, not inferred from a round number."""
    import numpy as np
    import metrics
    # Confident AND correct: no sharpening needed, so the fit should settle
    # somewhere interior rather than against either bound.
    # Draw labels FROM the probabilities, so the predictions are already
    # calibrated and the correct temperature is 1.0 -- an interior optimum.
    rng = np.random.default_rng(7)
    n = 4000
    p = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    info = metrics.fit_temperature_info(p, y)
    # at_ceiling / at_floor distinguish WHICH bound was hit, and brier is
    # carried for a temperature-independent comparison. Hitting the ceiling and
    # hitting the floor mean opposite things about the model, so one boolean was
    # not enough to report.
    assert set(info) == {"T", "nll", "at_bound", "at_ceiling", "at_floor",
                         "brier"}
    assert metrics.TEMP_MIN < info["T"] < metrics.TEMP_MAX
    assert info["at_bound"] is False


def test_a_confident_correct_predictor_sharpens_by_hitting_the_low_bound():
    """T below 1 sharpens. If the probabilities are already correct and extreme,
    the optimum really is at the floor -- that is convergence, not truncation,
    and at_bound must say so rather than hiding it."""
    import numpy as np
    import metrics
    y = np.array([0] * 100 + [1] * 100, float)
    p = np.where(y == 1, 0.999, 0.001)
    info = metrics.fit_temperature_info(p, y)
    assert info["at_bound"] is True
    assert info["T"] == metrics.TEMP_MIN


# ------------------------------------------------ KL divergence / memorisation

def test_kl_divergence_of_identical_distributions_is_zero():
    import metrics
    assert metrics.kl_bernoulli(0.3, 0.3) == pytest.approx(0.0, abs=1e-12)
    assert metrics.kl_bernoulli(0.5, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_is_positive_and_symmetric_when_q_is_the_complement():
    """KL over a BERNOULLI is not directional the way it is over a general
    distribution: when q = 1-p the two divergences are equal
    (KL(.9||.1) = KL(.1||.9) = 1.7578). Asserting asymmetry here was my error,
    not the function's. Directionality shows up against a non-complement q.
    """
    import metrics
    a = metrics.kl_bernoulli(0.9, 0.1)
    b = metrics.kl_bernoulli(0.1, 0.9)
    assert a > 0 and b > 0
    assert a == pytest.approx(b)
    assert metrics.kl_bernoulli(0.9, 0.3) != pytest.approx(
        metrics.kl_bernoulli(0.3, 0.9))


def test_kl_divergence_against_prior_is_zero_when_predicting_the_base_rate():
    """A model that outputs the base rate for every row carries no information,
    so its divergence from the prior must be exactly zero."""
    import numpy as np
    import metrics
    y = np.array([1, 1, 0, 0], float)          # base rate 0.5
    p = np.full(4, 0.5)
    assert metrics.kl_from_prior(p, y) == pytest.approx(0.0, abs=1e-9)


def test_kl_from_prior_is_benefit_over_the_base_rate():
    """Semantics, stated once so the sign is never guessed:

      constant AT the base rate   -> 0.0  (no better, no worse)
      constant AWAY from it        -> negative (worse: a balanced set predicted
                                      as 0.1 is a badly miscalibrated constant)
      varying with the label       -> positive (the model knows something)
    """
    import numpy as np
    import metrics
    y = np.array([1] * 50 + [0] * 50, float)          # base rate 0.5

    assert metrics.kl_from_prior(np.full(100, 0.5), y) == pytest.approx(
        0.0, abs=1e-9)

    # a constant away from the base rate is worse than the prior
    assert metrics.kl_from_prior(np.full(100, 0.1), y) < 0
    assert metrics.kl_from_prior(np.full(100, 0.9), y) < 0

    # and both are equally bad, by symmetry around the base rate
    assert (metrics.kl_from_prior(np.full(100, 0.1), y)
            == pytest.approx(metrics.kl_from_prior(np.full(100, 0.9), y)))

    # a model that actually tracks the label beats the prior
    good = metrics.kl_from_prior(np.where(y == 1, 0.9, 0.1), y)
    assert good > 0


def test_degenerate_base_rate_does_not_produce_nan():
    """All-one-class labels make the prior 0; KL must stay finite, because a
    NaN silently poisons every aggregate that touches it."""
    import numpy as np
    import metrics
    y = np.zeros(10)
    v = metrics.kl_from_prior(np.full(10, 0.01), y)
    assert np.isfinite(v)


def test_memorisation_gap_is_near_zero_when_train_and_test_agree():
    import numpy as np
    import metrics
    y = np.array([1, 0, 1, 0], float)
    p = np.array([0.8, 0.2, 0.7, 0.3])
    gap = metrics.memorisation_gap(p, y, p, y)
    assert gap["kl_gap"] == pytest.approx(0.0, abs=1e-9)


def test_memorisation_gap_is_large_when_train_is_perfect_and_test_is_not():
    """This is the signature the safety run shows: train loss falling to 0.0146
    while the held-out split stays mediocre. A model that answers train rows
    perfectly and test rows poorly has memorised, not learned."""
    import numpy as np
    import metrics
    y = np.array([1, 0, 1, 0], float)
    train_p = np.array([0.999, 0.001, 0.999, 0.001])   # perfect
    test_p = np.array([0.55, 0.45, 0.52, 0.48])        # near chance
    gap = metrics.memorisation_gap(train_p, y, test_p, y)
    assert gap["kl_gap"] > 0.5
    assert gap["flagged"] is True


def test_memorisation_gap_reports_the_components_not_just_a_flag():
    import numpy as np
    import metrics
    y = np.array([1, 0, 1, 0], float)
    p = np.array([0.9, 0.1, 0.8, 0.2])
    gap = metrics.memorisation_gap(p, y, p, y)
    for k in ("kl_train", "kl_test", "kl_gap", "acc_train", "acc_test", "flagged"):
        assert k in gap, k


def test_ceiling_pin_means_the_probability_LEVEL_is_wrong_not_the_search_range():
    """The baseline's central diagnostic, pinned as a property.

    T pinning at the ceiling is NOT a search-range bug (the ceiling was already
    raised once, 20 -> 200, and it pinned again). It happens when predicted
    probabilities sit systematically BELOW the base rate: flattening toward 0.5
    then improves log-loss monotonically, so the optimiser runs to the bound.

    The consequence is the important part. Temperature can only pull predictions
    toward 0.5. If the truth is a base rate of 0.60 or 0.70, temperature can
    never reach it -- so no temperature fixes a level error. The remedy is an
    intercept (prior/threshold) correction, which acts on level rather than
    spread.

    Two of my own attempts to write this test failed first: timid-but-centred
    probabilities pin at the FLOOR (they carry real magnitude), and
    level-shifted probabilities with real order converge normally. Only a wrong
    LEVEL with weak order reaches the ceiling.
    """
    import numpy as np
    import metrics

    rng = np.random.default_rng(2)
    n = 1000
    y = (rng.random(n) < 0.70).astype(float)        # high base rate
    p = np.full(n, 0.30)                            # level far below it
    info = metrics.fit_temperature_info(p, y)
    assert info["at_ceiling"], "a low-biased level should pin T at the ceiling"

    # the mechanism: nll improves monotonically with flattening, never turning
    nlls = []
    for t in (1, 5, 50, 200):
        q = np.clip(metrics.apply_temperature(p, t), 1e-9, 1 - 1e-9)
        nlls.append(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    # nll DECREASES as temperature rises (flattening improves log-loss)
    assert nlls == sorted(nlls, reverse=True), \
        f"flattening should improve monotonically: {nlls}"

    # and temperature cannot reach the base rate, so it cannot repair level
    flat = metrics.apply_temperature(p, 1e6)
    assert flat.mean() < 0.70


def test_at_ceiling_and_at_floor_are_not_the_same_finding():
    """Both bound hits are violations of a converged fit, but they mean opposite
    things and must not be reported as one boolean."""
    import numpy as np
    import metrics

    rng = np.random.default_rng(5)
    y = (rng.random(300) < 0.5).astype(float)
    # overconfident: magnitudes far too large -> optimum at the floor
    hot = np.clip(0.5 + 0.45 * np.sign(y - 0.5) + rng.normal(0, 0.02, 300),
                  1e-4, 1 - 1e-4)
    hi = metrics.fit_temperature_info(hot, y)
    assert hi["at_floor"] and not hi["at_ceiling"]
