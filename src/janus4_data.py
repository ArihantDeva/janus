"""Corpus loading for the Glimmer Janus fine-tune: rows -> prompts + 4-targets.

Split out of train_glimmer_janus.py so the data contract is importable and
testable without torch: nothing here needs a GPU, only the corpus file.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import janus3 as J3  # noqa: E402
import janus_interface as JI  # noqa: E402
import relabel_retry_wait as RR  # noqa: E402


def load_rows(path):
    """Corpus rows -> (prompt, 4-target, label_idx, group)."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    RR.relabel(rows)
    out = []
    for r in rows:
        label = r.get("janus_label", r.get("teacher_argmax"))
        if label not in J3.WORDS:
            continue  # null/unknown labels carry no target; never train on them
        st = JI.build_state(tool=r.get("tool"), arguments=r.get("arguments"),
                            error=r.get("error"))
        if label == "retry_wait":
            tgt = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            t = r["teacher"]
            tgt = np.array([t["debug"], t["fix_input"], t["fix_call"], 0.0])
            tgt /= tgt.sum()
            label = r["teacher_argmax"]
        out.append((JI.render(st), tgt, J3.WORDS.index(label), r["group"]))
    return out


def split_groups(rows, frac=0.15, seed=0):
    """Group-disjoint train/eval. Overlap raises; it must never happen."""
    groups = sorted({r[3] for r in rows})
    idx = np.random.default_rng(seed).permutation(len(groups))
    n_eval = max(1, int(round(frac * len(groups))))
    eg = {groups[i] for i in idx[:n_eval]}
    tr = [r for r in rows if r[3] not in eg]
    ev = [r for r in rows if r[3] in eg]
    assert not ({r[3] for r in tr} & eg), "group overlap between splits"
    return tr, ev


def save_result(out, seed, res):
    """Eval JSON beside the adapter. The artifact the submission cites."""
    import json
    import os
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/glimmer_janus_seed{seed}.json", "w"),
              indent=2)


def balance(tr):
    """Label counts over the train split, for the launch log."""
    import collections
    return dict(collections.Counter(J3.WORDS[y] for _, _, y, _ in tr))


def class_weights(counts):
    """Inverse-frequency weights, mean-normalized so LR needs no rescaling."""
    counts = np.asarray(counts, dtype=float)
    w = 1.0 / np.maximum(counts, 1.0)
    return w / w.mean()


def metrics4(P, y):
    """Accuracy, per-class recall, and ECE over the argmax confidence."""
    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == y).astype(float)
    ece = 0.0
    for b in range(10):
        m = (conf >= b / 10) & (conf < (b + 1) / 10)
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    per = {}
    for k, c in enumerate(J3.WORDS):
        m = y == k
        if m.sum():
            per[c] = (int(m.sum()), float((pred[m] == k).mean()))
    return dict(acc=float((pred == y).mean()), ece=float(ece), per=per)
