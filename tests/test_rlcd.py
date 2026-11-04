"""Tests for the distillation and RLCD losses.

These are pure functions, so every property that makes them meaningful is checked
directly. The recurring theme: a loss can be numerically fine and semantically
empty, and the empty version is the one that gets shipped.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import rlcd as R

CL = ("debug", "fix_input", "fix_call")


# ---------------------------------------------------------------------------
# distillation
# ---------------------------------------------------------------------------

def test_soft_ce_and_kl_agree_up_to_the_target_entropy():
    """Two functions, one objective. If they disagreed, the reported number would
    depend on which one happened to be called."""
    logits = np.array([[2.0, 0.5, -1.0]])
    t = np.array([[0.7, 0.2, 0.1]])
    ce = R.soft_ce(logits, t)
    kl = R.kl_div(logits, t)
    # soft_ce = kl + H(target)
    h = -float((t * np.log(t)).sum())
    assert ce == pytest.approx(kl + h, abs=1e-9)


def test_kl_is_zero_when_student_equals_teacher():
    """The minimum of the objective must actually be attained."""
    p = np.array([[0.6, 0.3, 0.1]])
    logits = np.log(p)
    assert R.kl_div(logits, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_penalises_a_confident_wrong_answer_more_than_a_hedged_one():
    """Calibration is the point: a confident wrong answer must cost more."""
    t = np.array([[1.0, 0.0, 0.0]])
    confident_wrong = np.array([[0.0, 5.0, 0.0]])
    hedged = np.array([[0.0, 0.0, 0.0]])
    assert R.kl_div(confident_wrong, t) > R.kl_div(hedged, t)


def test_temperature_one_is_the_identity():
    d = np.array([0.7, 0.2, 0.1])
    assert R.soften(d, 1.0) == pytest.approx(d, abs=1e-9)


def test_temperature_is_load_bearing_in_both_directions():
    """T is not decoration. Above 1 flattens, below 1 sharpens; if either direction
    did nothing the knob would be inert and the near-zero teacher entries -- where
    Janus's knowledge actually lives -- would carry no gradient."""
    d = np.array([0.94, 0.05, 0.01])
    hotter = R.soften(d, 2.0)
    colder = R.soften(d, 0.5)
    assert hotter[0] < d[0] and hotter.min() > d.min(), hotter
    assert colder[0] > d[0] and colder.min() < d.min(), colder
    assert colder.shape == d.shape


def test_soften_renormalises_and_rejects_nonsense():
    out = R.soften(np.array([2.0, 2.0, 2.0]), 1.0)      # unnormalised input
    assert out.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        R.soften(np.array([0.5, 0.5, 0.0]), 0.0)
    with pytest.raises(ValueError):
        R.soften(np.array([0.5, -0.1, 0.6]), 1.0)


def test_near_one_hot_teacher_gains_from_temperature():
    """Measured fact: Janus's median confidence is 0.97, so its minority classes sit
    at ~0.02 and carry almost no gradient at T=1.

    The actual ratio measured here is ~5.8x (0.0200 -> 0.1153 on the runner-up).
    My first version asserted >20x, a number I invented rather than measured; the
    test failed and the code was right. Pinning the measured ratio rather than a
    hoped-for one is the whole point of writing the test.
    """
    d = np.array([0.97, 0.02, 0.01])
    t1 = R.soften(d, 1.0)[1]
    t2 = R.soften(d, 2.0)[1]
    assert t1 == pytest.approx(0.02, abs=1e-9)
    assert t2 == pytest.approx(0.1153, abs=1e-3)
    assert t2 > 5 * t1


# ---------------------------------------------------------------------------
# RLCD
# ---------------------------------------------------------------------------

def test_contrastive_margin_is_monotone_in_the_gap():
    pos, neg = np.array([3.0]), np.array([1.0])
    far_apart = R.contrastive_margin(pos, neg, 0, margin=0.5)
    close = R.contrastive_margin(np.array([3.0]), np.array([2.9]), 0, margin=0.5)
    assert far_apart == 0.0, "a satisfied margin must not keep pushing"
    assert close > far_apart


def test_contrastive_margin_hinges_at_zero():
    """No drift once satisfied: a hinge that never reaches zero keeps moving the
    representation for no gain."""
    assert R.contrastive_margin(np.array([10.0]), np.array([0.0]), 0, margin=0.5) == 0.0


def test_pairs_use_the_runner_up_not_a_random_class():
    """The negative must be the teacher's second choice. A random class makes the
    task free and the loss collapses to zero without teaching anything."""
    d = np.array([0.7, 0.25, 0.05])
    pos, neg = R.contrastive_pairs(d, np.random.default_rng(0), CL)
    assert pos == 0 and neg == 1


def test_pairs_are_none_when_the_teacher_has_no_runner_up():
    """A (near-)uniform teacher drew no distinction, so inventing one would teach
    a boundary the teacher never asserted."""
    assert R.contrastive_pairs(np.array([0.34, 0.33, 0.33]), np.random.default_rng(0), CL) is None
    assert R.contrastive_pairs(np.array([0.2, 0.2, 0.2]), np.random.default_rng(0), CL) is None
    # a 1% gap is noise, not a distinction; sensitivity check on the threshold itself
    assert R.contrastive_pairs(np.array([0.36, 0.35, 0.29]), np.random.default_rng(0), CL) is None
    assert R.contrastive_pairs(np.array([0.60, 0.35, 0.05]), np.random.default_rng(0), CL) == (0, 1)


def test_pairs_reject_a_wrong_class_count():
    with pytest.raises(ValueError):
        R.contrastive_pairs(np.array([0.5, 0.5]), np.random.default_rng(0), CL)


def test_rlcd_strength_zero_is_exactly_plain_distillation():
    """The most important test here: if stage 2 silently equals stage 1, a run can
    report 'RLCD ran' while nothing changed."""
    s = np.array([1.0, 0.0, -1.0])
    t = np.array([0.6, 0.3, 0.1])
    assert R.rlcd_soft(s, t, 1, CL, strength=0.0) == pytest.approx(R.kl_div(s, t))


def test_rlcd_penalty_grows_as_the_wrong_class_gets_more_probable():
    t = np.array([0.6, 0.3, 0.1])
    low_wrong = R.rlcd_soft(np.array([2.0, -2.0, 0.0]), t, 1, CL, strength=1.0)
    high_wrong = R.rlcd_soft(np.array([2.0, 2.0, 0.0]), t, 1, CL, strength=1.0)
    assert high_wrong > low_wrong


def test_rlcd_rejects_nothing_and_handles_a_null_negative():
    """A row with no runner-up has wrong_idx None; that must degrade to plain
    distillation rather than raising mid-epoch."""
    s, t = np.array([1.0, 0.0, -1.0]), np.array([0.4, 0.3, 0.3])
    assert R.rlcd_soft(s, t, None, CL) == pytest.approx(R.kl_div(s, t))


def test_entropy_flags_an_uninformative_teacher_row():
    """Entropy is how the pipeline decides a teacher row carries no signal."""
    assert R.entropy(np.array([1.0, 0.0, 0.0])) < R.entropy(np.array([0.34, 0.33, 0.33]))
