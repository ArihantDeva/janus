"""Tests for the real-Janus extraction and scoring.

Every case here is a defect that actually occurred while building it, or a trap
that would silently invalidate a number:

  - option labels contain SPACES and COMMAS ("Prototype stage, pre-launch, no
    users, no revenue"); a \\w-only or comma-splitting pattern silently truncates
    them, which made the parsed label disagree with the stated choice on 2/44 rows
  - question ids contain '|' and ':' ("deposit:0|harmer"); excluding those
    characters silently dropped 19 real decisions
  - Janus's renderer lists the CHOSEN option first, so argmax measures the format
    rather than a preference. Agreement is therefore not evidence, and it needs
    an explicit first-option null to be interpretable.
  - decisions inside one call share a state and are not independent, so
    intervals must be bootstrapped over calls
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_janus_live import (  # noqa: E402
    batch_bootstrap, brier, score_predictions, total_variation,
)
from janus_live import parse_result  # noqa: E402


def test_parses_noul_line():
    out, dropped = parse_result("correctness: yes 0.92  <- Does this work?")
    assert dropped == 0
    d = out[0]
    assert d["type"] == "noul"
    assert d["question"] == "Does this work?"
    # "yes 0.92" means P(yes)=0.92, so the pair is [0.08, 0.92]
    assert d["probs"][1] == pytest.approx(0.92)
    assert d["probs"][0] == pytest.approx(0.08)


def test_parses_choice_with_comma_and_space_labels():
    """The measured defect: labels are prose, not identifiers."""
    line = ("route: Prototype stage, pre-launch, no users, no revenue (conf 1.00) "
            "[Prototype stage, pre-launch, no users, no revenue 1.00, "
            "Growth stage, paying users 0.00, Enterprise, multiple teams 0.00]  "
            "<- Which stage?")
    out, _ = parse_result(line)
    d = out[0]
    assert len(d["labels"]) == 3
    assert d["labels"][0] == "Prototype stage, pre-launch, no users, no revenue"
    assert d["labels"][1] == "Growth stage, paying users"
    assert d["chosen"] == "Prototype stage, pre-launch, no users, no revenue"
    assert abs(sum(d["probs"]) - 1.0) < 1e-6


def test_parses_ids_containing_pipe_and_colon():
    """Ids like 'deposit:0|harmer' are real; excluding those chars dropped 19
    decisions, a quarter of the dataset."""
    out, dropped = parse_result(
        "deposit:0|harmer: yes 0.05  <- In candidate [deposit:0], is this a harm?")
    assert dropped == 0
    assert out[0]["id"] == "deposit:0|harmer"


def test_parses_score_line():
    out, _ = parse_result("scope: 1.78/3 (conf 0.73)  <- Is this minimal?")
    d = out[0]
    assert d["type"] == "score"
    assert d["value"] == pytest.approx(1.78)
    assert d["max"] == 3


def test_truncation_marker_is_dropped_not_guessed():
    """A truncated log line is not an answer."""
    out, dropped = parse_result("… [truncated 121 chars]")
    assert out == []
    assert dropped == 1


def test_total_variation_bounds():
    assert total_variation([1, 0], [1, 0]) == pytest.approx(0.0)
    assert total_variation([1, 0], [0, 1]) == pytest.approx(1.0)
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_bootstrap_widens_with_grouping():
    """Grouped intervals must be wider than per-item ones when items within a
    group are correlated. If this ever inverts, the dependence structure in the
    data changed and the intervals need re-deriving."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 4)
    # strong within-group correlation
    per_group = rng.uniform(0, 1, 20)
    values = per_group[groups]

    lo_g, hi_g = batch_bootstrap(values, groups, n_boot=500)
    lo_i, hi_i = batch_bootstrap(values, np.arange(len(values)), n_boot=500)
    assert (hi_g - lo_g) > (hi_i - lo_i), "grouped interval should be wider"


def test_unsupported_decisions_are_counted_not_scored_as_wrong():
    """A model with fewer options than a question must not be punished for a
    label-set mismatch -- that would inflate the gap for a reason unrelated to
    quality."""
    decisions = [
        {"type": "choice", "probs": [0.7, 0.2, 0.1], "labels": ["a", "b", "c"]},
        {"type": "choice", "probs": [0.6, 0.4], "labels": ["x", "y"]},
    ]

    def only_two_options(d):
        if len(d["probs"]) != 2:
            return None
        return [0.5, 0.5], d["labels"]

    s = score_predictions(decisions, only_two_options)
    assert s["unsupported"] == 1
    assert s["n"] == 1


def test_first_option_null_scores_perfect_agreement_by_construction():
    """Documents why agreement alone is not evidence on this data."""
    decisions = [{"type": "choice", "probs": [0.9, 0.1], "labels": ["a", "b"],
                  "ts": "t"}]

    def first(_d):
        return [1.0, 0.0], _d["labels"]

    s = score_predictions(decisions, first)
    assert s["agreement"] == 1.0
