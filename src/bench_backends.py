"""Score the classifier backends on held-out rows.

TIES ARE HANDLED EXPLICITLY, because getting this wrong corrupts the entire
comparison. Found by a contradiction: the keyword null was labelled `retry` on 559
fix_call rows, yet only 19 of them contained any retry keyword.

Cause: when no keyword matches, the null returns a genuinely UNIFORM distribution.
`max(dist, key=dist.get)` then returns the first key in insertion order, which is
always "retry". So every unmatched row was silently called `retry` -- the metric
was reading dict ordering, not the classifier. This is the same defect class as the
seed-disagreement tie-shuffle in run 29: tied scores + argmax measures argsort.

POLICY: a tied top score is an ABSTENTION, not a prediction. It is excluded from
accuracy and reported separately as coverage. That is the only treatment that
cannot invent a class the backend did not choose.
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import CLASSES, get_backend  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIE_EPS = 1e-9


def decide(dist):
    """(class, confidence, tied) -- `tied` means the top score is not unique."""
    top = max(dist.values())
    winners = [c for c, v in dist.items() if v >= top - TIE_EPS]
    return (winners[0] if len(winners) == 1 else None), top, len(winners) > 1


def score(name, fn, rows, ys):
    t0 = time.time()
    P = []
    for i, r in enumerate(rows):
        try:
            P.append([fn(r["x"])[c] for c in CLASSES])
        except Exception as e:
            print(f"  [{name}] row {i} failed: {type(e).__name__}: {e}"[:160])
            P.append([1 / len(CLASSES)] * len(CLASSES))
        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    P = np.array(P)

    dec = [decide(dict(zip(CLASSES, p))) for p in P]
    answered = [i for i, d in enumerate(dec) if d[0] is not None]
    pred = np.full(len(ys), -1)
    for i in answered:
        pred[i] = CLASSES.index(dec[i][0])

    conf = P.max(1)
    onehot = np.eye(len(CLASSES))[ys]
    brier = float(((P - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(ys)), ys], 1e-12, None)).mean())

    ok = np.array([pred[i] == ys[i] for i in answered]) if answered else np.array([])
    acc = float(ok.mean()) if len(ok) else float("nan")
    cov = len(answered) / len(ys)

    correct = np.array([(pred[i] == ys[i]) if pred[i] >= 0 else 0.0
                        for i in range(len(ys))])
    ece, nb = 0.0, 0
    for b in range(10):
        m = (conf >= b / 10) & (conf < (b + 1) / 10)
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
            nb += 1

    # accuracy restricted to rows answered above each confidence bar
    def acc_at(bar):
        sel = [i for i in answered if conf[i] >= bar]
        return (float(np.mean([pred[i] == ys[i] for i in sel])), len(sel) / len(ys))

    a50, c50 = acc_at(0.5)
    a80, c80 = acc_at(0.8)

    per = {}
    for k in range(len(CLASSES)):
        sel = [i for i in answered if ys[i] == k]
        if sel:
            per[CLASSES[k]] = (len(sel), float(np.mean([pred[i] == ys[i] for i in sel])))

    return dict(name=name, n=len(ys), answered=len(answered), coverage=cov,
                acc=acc, brier=brier, nll=nll, ece=ece,
                acc_at_50=a50, cov_at_50=c50, acc_at_80=a80, cov_at_80=c80,
                per=per, secs=time.time() - t0)


def main():
    n = int(os.environ.get("N", "200"))
    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "data", "responses", "rule_held.jsonl"))]
    rows = [r for r in rows if r["y"] in CLASSES]
    random.seed(0)
    rows = random.sample(rows, min(n, len(rows)))
    ys = np.array([CLASSES.index(r["y"]) for r in rows])
    maj = collections.Counter(ys).most_common(1)[0][1] / len(ys)

    print(f"held-out rows: {len(rows)}   majority baseline: {100*maj:.1f}%")
    print("policy: a tied top score is an ABSTENTION, not a prediction\n")

    res = []
    for name in ("keyword", "janus"):
        try:
            res.append(score(name, get_backend(name), rows, ys))
        except Exception as e:
            print(f"{name} unavailable: {type(e).__name__}: {e}")

    print("\n" + "=" * 88)
    print(f"{'backend':9s} {'acc(all)':>9s} {'cover':>6s} {'acc@.5':>7s} {'cov@.5':>7s} "
          f"{'acc@.8':>7s} {'brier':>7s} {'ece':>6s}")
    print(f"{'majority':9s} {100*maj:8.1f}%")
    for r in res:
        print(f"{r['name']:9s} {100*r['acc']:8.1f}% {100*r['coverage']:5.1f}% "
              f"{100*r['acc_at_50']:6.1f}% {100*r['cov_at_50']:6.1f}% "
              f"{100*r['acc_at_80']:6.1f}% {r['brier']:7.4f} {r['ece']:6.4f}")

    print("\nper-class recall (answered rows only):")
    for r in res:
        print(f"  {r['name']:8s} " + "  ".join(
            f"{k}={100*v[1]:.0f}%(n{v[0]})" for k, v in r["per"].items()))

    out = os.path.join(ROOT, "data", "responses", "baselines.json")
    json.dump({"n": len(ys), "majority": maj, "tie_policy": "abstain",
               "results": res}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
