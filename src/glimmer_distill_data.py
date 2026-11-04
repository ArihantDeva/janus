"""Corpus loading + group-disjoint split for the Janus 4-class distill."""
from __future__ import annotations

import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.environ.get("JANUS_CORPUS",
                         os.path.join(ROOT, "data", "distill", "janus.jsonl"))

CLASSES = ("debug", "fix_input", "fix_call", "retry_wait")
_CAUSAL_ATTRS = ("is_causal", "causal", "_is_causal", "causal_mask")


def load_corpus(path=CORPUS, min_conf=0.0):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("teacher")
        if not isinstance(t, dict):
            continue
        if max(t.values()) < min_conf:
            continue
        r["_label"] = max(t, key=t.get)
        if r["_label"] not in CLASSES:
            continue
        rows.append(r)
    return rows


def split_by_group(rows, frac=0.15, seed=0):
    groups = sorted({r.get("group", "?") for r in rows})
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(groups))
    n_eval = max(1, int(round(frac * len(groups))))
    eval_g = {groups[i] for i in idx[:n_eval]}
    ev = [r for r in rows if r.get("group", "?") in eval_g]
    tr = [r for r in rows if r.get("group", "?") not in eval_g]
    overlap = {r.get("group", "?") for r in tr} & eval_g
    if overlap:
        raise AssertionError(f"group overlap: {sorted(overlap)[:5]}")
    return tr, ev, sorted(eval_g)


def teacher_target(r):
    t = r["teacher"]
    v = np.array([t.get(c, 0.0) for c in CLASSES], dtype=np.float64)
    return v / v.sum()


def encode_input(r, max_arg_chars=4000):
    args = r.get("arguments", {})
    arg_s = json.dumps(args, sort_keys=True, default=str)[:max_arg_chars]
    err = str(r.get("error", ""))[:4000]
    return (f"[tool: {r.get('tool', '?')}]\n[args: {arg_s}]\n{err}\n\n"
            f"question\nthe reason this tool call failed\n\nchoices\n"
            f"A) debug\nB) fix_input\nC) fix_call\nD) retry_wait")


