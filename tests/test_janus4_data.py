"""The Glimmer Janus data contract: rows -> prompts + 4-targets + disjoint split.

Runs on plain CPU python (no torch): the trainer must never be the first
thing to touch the corpus. Group-disjointness, 4-wide normalized targets,
and the retry_wait override are all pinned here.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import janus3 as J3  # noqa: E402
import janus4_data as JD  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "distill",
                       "janus.jsonl")


def test_load_rows_keep_only_labelled_rows_with_4_wide_targets():
    rows = JD.load_rows(CORPUS)
    assert len(rows) == 25311
    for prompt, tgt, y, group in rows:
        assert isinstance(prompt, str) and prompt.startswith("state\ntool:")
        assert tgt.shape == (4,) and abs(tgt.sum() - 1.0) < 1e-9
        assert 0 <= y <= 3 and J3.WORDS[y] in J3.WORDS
        assert group


def test_split_is_group_disjoint():
    rows = JD.load_rows(CORPUS)
    tr, ev = JD.split_groups(rows)
    assert len(tr) > 20000 and len(ev) > 3000
    assert not ({r[3] for r in tr} & {r[3] for r in ev})


def test_retry_wait_rows_are_one_hot_and_auditable():
    rows = JD.load_rows(CORPUS)
    rw = [(p, t, y, g) for p, t, y, g in rows if y == 3]
    assert len(rw) == 954
    for _, tgt, _, _ in rw:
        assert list(tgt) == [0.0, 0.0, 0.0, 1.0]


def test_metrics4_is_exact_on_oracle():
    rows = JD.load_rows(CORPUS)
    _, ev = JD.split_groups(rows)
    y = np.array([yy for _, _, yy, _ in ev])
    m = JD.metrics4(np.eye(4)[y], y)
    assert m["acc"] == 1.0 and m["ece"] == 0.0
    assert set(m["per"]) == set(J3.WORDS)
