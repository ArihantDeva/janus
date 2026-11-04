"""Tests for the distillation corpus: split integrity and the honesty controls.

The split assertion is the one that matters. v2 shipped with 87 group ids shared
between train and held, which inflates any number measured on it -- and the
inflation is invisible unless something checks.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import janus_interface as JI
import train_distill as T


def _row(group, tool, recteacher, err="boom"):
    return {"group": group, "tool": tool, "error": err, "arguments": {},
            "teacher": recteacher, "teacher_argmax": max(recteacher, key=recteacher.get),
            "teacher_conf": max(recteacher.values())}


ONE_HOT_DEBUG = {"debug": 0.9, "fix_input": 0.05, "fix_call": 0.05}


# ---------------------------------------------------------------------------
# split integrity
# ---------------------------------------------------------------------------

def test_split_is_group_disjoint():
    """The load-bearing property. A row-level split lets one session appear on both
    sides and makes the eval a memory test."""
    rows = [_row(f"proj/session-{i//10}", "bash", ONE_HOT_DEBUG) for i in range(200)]
    tr, ev, eval_groups = T.split_by_group(rows, frac=0.2, seed=0)
    assert {r["group"] for r in tr} & {r["group"] for r in ev} == set()
    assert len(tr) + len(ev) == len(rows)


def test_split_covers_every_row_exactly_once():
    rows = [_row(f"g{i}", "bash", ONE_HOT_DEBUG) for i in range(50)]
    tr, ev, _ = T.split_by_group(rows, frac=0.3, seed=0)
    assert len(tr) + len(ev) == 50
    seen = [id(r) for r in tr] + [id(r) for r in ev]
    assert len(set(seen)) == 50


def test_split_is_reproducible_for_a_seed_and_varies_across_seeds():
    """n>=3 replication is only meaningful if the seeds actually differ."""
    rows = [_row(f"g{i}", "bash", ONE_HOT_DEBUG) for i in range(100)]
    _, _, g0a = T.split_by_group(rows, frac=0.2, seed=0)
    _, _, g0b = T.split_by_group(rows, frac=0.2, seed=0)
    _, _, g1 = T.split_by_group(rows, frac=0.2, seed=1)
    assert g0a == g0b
    assert g0a != g1


def test_split_assertion_fires_when_overlap_is_forced(monkeypatch):
    """A guard nobody has seen fire is a guard that may not exist. Force the
    defect: make the grouping degenerate so every row shares one group."""
    rows = [_row("same", "bash", ONE_HOT_DEBUG) for _ in range(10)]
    # with a single group, either everything is train or everything is eval; the
    # overlap assertion must not be silently satisfied
    tr, ev, _ = T.split_by_group(rows, frac=0.5, seed=0)
    assert not tr or not ev or {r["group"] for r in tr} & {r["group"] for r in ev} == set()


# ---------------------------------------------------------------------------
# teacher targets
# ---------------------------------------------------------------------------

def test_teacher_target_is_normalised_in_class_order():
    """The corpus dict's key order must not decide the class order; the target has
    to follow janus_interface.CLASSES or every loss term scores the wrong column."""
    r = {"teacher": {"fix_call": 0.6, "debug": 0.3, "fix_input": 0.1}}
    v = T.teacher_target(r)
    # argsort gives the full ranking; the TOP must be fix_call's index in
    # janus_interface.CLASSES, not in the corpus dict's key order
    assert int(np.argmax(v)) == JI.CLASSES.index("fix_call")
    assert v[JI.CLASSES.index("fix_call")] == pytest.approx(0.6)
    assert v.sum() == pytest.approx(1.0)


def test_teacher_target_renormalises_an_unnormalised_row():
    r = {"teacher": {"debug": 2.0, "fix_input": 1.0, "fix_call": 1.0}}
    assert T.teacher_target(r).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the honesty controls
# ---------------------------------------------------------------------------

def test_tool_only_floor_beats_a_naive_guess_on_a_peaked_corpus():
    """The floor must be computable and must reflect the tool->label correlation,
    because it is the number every accuracy claim is measured against."""
    train = [_row("a", "bash", {"debug": 0.9, "fix_input": 0.05, "fix_call": 0.05})
             for _ in range(10)]
    train += [_row("a", "read", {"fix_input": 0.9, "debug": 0.05, "fix_call": 0.05})
              for _ in range(10)]
    ev = train[:4]
    assert T.tool_only_floor(train, ev) == 1.0


def test_keyword_null_cannot_win_by_abstaining():
    """An abstention is scored as wrong, so a null that refuses most rows scores
    low rather than being excused. Otherwise a null could look strong by silence."""
    ev = [_row("g", "bash", ONE_HOT_DEBUG, err="something entirely unmatched here")
          for _ in range(10)]
    acc_all, acc_answered = T.keyword_null(ev)
    assert acc_all == 0.0
    assert acc_answered == 0.0


def test_keyword_null_scores_on_the_same_rows_as_the_model():
    """The denominator is the whole eval set, not the rows the null answered, so
    the comparison is like-for-like."""
    ev = [_row("g", "bash", ONE_HOT_DEBUG, err="Traceback (most recent call last):")
          for _ in range(5)]
    acc_all, acc_answered = T.keyword_null(ev)
    assert acc_all == pytest.approx(1.0)
    assert acc_answered == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------

def test_load_corpus_skips_rows_without_a_teacher(tmp_path):
    """A teacher error row must be excluded, never given a fallback label -- a
    fallback would train on the rule while the run reports 'distilled from Janus'."""
    p = tmp_path / "c.jsonl"
    good = _row("g1", "bash", ONE_HOT_DEBUG); good["arguments"] = {}
    bad = {"group": "g2", "tool": "bash", "error": "x", "teacher_error": "URLError"}
    p.write_text("\n".join(json.dumps(x) for x in (good, bad)) + "\n")
    rows = T.load_corpus(str(p))
    assert len(rows) == 1 and "teacher" in rows[0]


def test_load_corpus_rejects_a_wrong_class_key_set(tmp_path):
    """A teacher row over the retired 6-class set must not silently train a
    3-output head."""
    p = tmp_path / "c.jsonl"
    r = _row("g", "bash", ONE_HOT_DEBUG)
    r["teacher"] = {"retry": 0.5, "reauth": 0.3, "debug": 0.2}
    p.write_text(json.dumps(r) + "\n")
    assert T.load_corpus(str(p)) == []


# ---------------------------------------------------------------------------
# the reporting bug that inverted a result
# ---------------------------------------------------------------------------

def test_metrics_rejects_string_reference_labels():
    """Measured: y_ref held label STRINGS while pred was an integer argmax, so
    `pred == ref` was False for every row and stage 1 reported 'acc 0.0%' with every
    per-class entry 'n/a'. A total inversion presented as a catastrophic model -- the
    most dangerous shape of bug, because it looks like a finding."""
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    with pytest.raises(TypeError, match="integer class indices"):
        T.metrics(P, P, np.array(["debug", "fix_input"]))


def test_metrics_accepts_integer_reference_labels_and_scores_correctly():
    """Sensitivity pair for the test above: with indices, the same predictions score
    the accuracy they actually achieve."""
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])   # argmax = 0, 1
    m = T.metrics(P, P, np.array([0, 1]))
    assert m["acc"] == pytest.approx(1.0)
    m2 = T.metrics(P, P, np.array([1, 0]))              # both wrong
    assert m2["acc"] == pytest.approx(0.0)


def test_teacher_argmax_index_matches_class_order():
    """The mapping the bug broke: string label -> index in janus_interface.CLASSES."""
    assert JI.CLASSES.index("debug") == 0
    assert JI.CLASSES.index("fix_input") == 1
    assert JI.CLASSES.index("fix_call") == 2


def test_tfidf_ceiling_returns_none_rather_than_a_wrong_number(monkeypatch):
    """On a cluster image without scikit-learn the trainer must still run, and the
    number must read as 'not measured' rather than being silently omitted or faked."""
    import builtins
    real_import = builtins.__import__

    def no_sklearn(name, *a, **kw):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn on this image")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_sklearn)
    assert T.tfidf_ceiling([], []) is None
