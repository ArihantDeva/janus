"""Is the tool-outcome dataset learnable, or is it noise?

ONLY THREE NUMBERS MATTER HERE, and all three are nulls a real model must beat:

  majority     predict "success" for everything. 96% of rows are successes, so
               any accuracy figure is meaningless unless it is stated against
               this. This is the same trap as the safety benchmarks, where a
               constant predictor sat exactly on the floor for 30 runs.
  shortcut     a lexical rule with no learning: the outcome of a command is
               strongly hinted by its text ("python: command not found"). If a
               bag-of-words rule gets most of the signal, a transformer adds
               little and the dataset is mostly about vocabulary.
  tool_only    predict from the tool name alone. Cheap, and it shows how much is
               just "bash fails more than read".

If a trained model cannot beat `shortcut`, there is nothing beyond surface text
and the dataset should not be used to claim discrimination.

SPLIT IS BY SESSION GROUP, never by row. Measured: 1078 failure texts repeat
across sessions, one appearing 551 times, so a row split would put identical
commands on both sides and report a fake score.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "toolout", "calls.jsonl")


def load():
    rows = [json.loads(l) for l in open(PATH)]
    X = [r["x"] for r in rows]
    y = [r["y"] for r in rows]
    g = [r["group"] for r in rows]
    return X, np.array(y), np.array(g)


def split_by_group(g, seed=0, frac=0.2):
    """Held-out groups, so no session contributes to both sides."""
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    cut = int(len(uniq) * (1 - frac))
    tr_g, te_g = set(uniq[:cut]), set(uniq[cut:])
    tr = np.array([i for i, x in enumerate(g) if x in tr_g])
    te = np.array([i for i, x in enumerate(g) if x in te_g])
    return tr, te


def majority_baseline(y_tr, y_te):
    vals, counts = np.unique(y_tr, return_counts=True)
    top = vals[counts.argmax()]
    return float((y_te == top).mean()), str(top)


def tool_only_baseline(tools_tr, y_tr, tools_te, y_te):
    """P(label) = per-tool majority class seen in train."""
    best = {}
    for t, y in zip(tools_tr, y_tr):
        best.setdefault(t, {})
        best[t][y] = best[t].get(y, 0) + 1
    pred = []
    for t in tools_te:
        d = best.get(t)
        pred.append(max(d, key=d.get) if d else "success")
    return float((np.array(pred) == y_te).mean())


def shortcut_baseline(X_tr, y_tr, X_te, y_te):
    """Bag-of-words logistic regression. No deep model, no tuning."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          min_df=3, sublinear_tf=True)
    A = vec.fit_transform(X_tr)
    B = vec.transform(X_te)
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(A, y_tr)
    pred = clf.predict(B)
    acc = float((pred == y_te).mean())

    # per-class recall on the rare classes, which is where the signal lives
    per = {}
    for c in np.unique(y_te):
        m = y_te == c
        if m.sum():
            per[str(c)] = float((pred[m] == c).mean())
    return acc, per


def binary_view(y):
    """failed / not, for a headline number."""
    return np.where(y == "success", "success", "failed")


def main():
    if not os.path.exists(PATH):
        raise SystemExit(f"{PATH} missing; run src/tool_outcomes.py")
    X, y, g = load()
    print(f"rows {len(y)}  groups {len(np.unique(g))}")
    print("class balance:", dict(zip(*np.unique(y, return_counts=True))))

    tr, te = split_by_group(g)
    print(f"\nsplit by SESSION: train {len(tr)} rows / {len(set(g[tr]))} groups, "
          f"test {len(te)} rows / {len(set(g[te]))} groups")
    overlap = set(g[tr]) & set(g[te])
    print(f"group overlap between splits: {len(overlap)} (must be 0)")

    Xtr = [X[i] for i in tr]; Xte = [X[i] for i in te]
    ytr, yte = y[tr], y[te]

    maj, cls = majority_baseline(ytr, yte)
    print(f"\n{'majority':22s} {maj*100:5.1f}%   (always {cls!r})")

    tools = [json.loads("{}") if False else None for _ in ()]  # noqa: F841
    tl_tr = [t for t in _tools(tr)]
    tl_te = [t for t in _tools(te)]
    print(f"{'tool-only':22s} {tool_only_baseline(tl_tr, ytr, tl_te, yte)*100:5.1f}%")

    acc, per = shortcut_baseline(Xtr, ytr, Xte, yte)
    print(f"{'lexical (tfidf+LR)':22s} {acc*100:5.1f}%")
    print("  per-class recall:", {k: round(v, 2) for k, v in per.items()})

    # binary headline
    btr, bte = binary_view(ytr), binary_view(yte)
    bmaj, _ = majority_baseline(btr, bte)
    bacc, bper = shortcut_baseline(Xtr, btr, Xte, bte)
    print(f"\nbinary failed/success:")
    print(f"  {'majority':20s} {bmaj*100:5.1f}%")
    print(f"  {'lexical':20s} {bacc*100:5.1f}%")
    print(f"  per-class recall: {bper}")


def _tools(idx):
    for i in idx:
        yield _TOOLS[i]


_TOOLS = []


if __name__ == "__main__":
    _X, _y, _g = None, None, None
    if os.path.exists(PATH):
        _rows = [json.loads(l) for l in open(PATH)]
        _TOOLS.extend(r["tool"] for r in _rows)
    main()
