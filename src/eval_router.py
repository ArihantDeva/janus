"""Measure the router: accuracy vs escalation rate vs cost, on held-out sessions.

THE ONLY CLAIM WORTH MAKING
---------------------------
Not "the local model is accurate". It is a 1B model on a keyword-shaped task and
a TF-IDF classifier scores 92.2% on the same rows. The claim is narrower and
actually defensible:

    a calibrated small model can decide MOST failures locally, and its own
    confidence identifies the rest well enough to route them to a frontier model,
    so the expensive model is called on a minority of rows without giving up
    accuracy.

That claim is measurable and could be false. Two ways it fails:
  * the confidence is uninformative -> escalation saves nothing, because the
    escalated rows are no harder than the rest
  * tier 1 is so weak that even with escalation the accuracy stays near the
    majority baseline

Both are reported. The output is a curve, not a single operating point, because
picking one threshold and quoting its accuracy is how you fit a knob to a test
set.

COST MODEL
----------
Frontier cost is measured, not estimated: every response reports its reasoning
and total tokens. Local cost is zero by construction (it runs on the machine), so
the meaningful ratio is frontier calls per 100 failures, plus the tokens those
calls burned.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from janus_route import CLASSES, route  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    p = os.path.join(ROOT, "data", "responses", f"{name}.jsonl")
    return [json.loads(l) for l in open(p)]


def auc_uncertainty(conf, correct):
    """AUC of the confidence as a predictor of correctness.

    0.5 means confidence carries no information and escalation is pointless.
    Computed on rank because confidences are heavily tied.
    """
    pairs = sorted(zip(conf, correct))
    n1 = sum(correct)
    n0 = len(correct) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    rank, i, s = 1, 0, 0.0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (rank + (rank + (j - i) - 1)) / 2
        s += avg * sum(1 for k in range(i, j) if pairs[k][1])
        rank += j - i
        i = j
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="rule_held")
    ap.add_argument("--base", default=os.environ.get("JANUS_BASE_MODEL",
                                                     "unsloth/Llama-3.2-1B"))
    ap.add_argument("--adapter", default=os.environ.get("JANUS_ADAPTER"))
    ap.add_argument("--limit", type=int, default=0,
                    help="frontier-call cap; 0 means escalate greedily per threshold")
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()

    rows = [r for r in load(a.split) if r["y"] in CLASSES]
    print(f"split {a.split}: {len(rows)} rows, classes {CLASSES}")

    from janus_route import LocalTier
    local = LocalTier(a.base, a.adapter)

    # one local pass, reused across every threshold -- the frontier is the only
    # part that costs money, so it is called at most once per row
    local_pred, local_conf, ys = [], [], []
    for i, r in enumerate(rows):
        p = local(r["x"])
        best = max(p, key=p.get)
        local_pred.append(best)
        local_conf.append(p[best])
        ys.append(r["y"])
        if (i + 1) % 200 == 0:
            print(f"  local {i+1}/{len(rows)}", flush=True)

    acc = sum(p == y for p, y in zip(local_pred, ys)) / len(ys)
    maj = collections.Counter(ys).most_common(1)[0][1] / len(ys)
    print(f"\ntier 1 alone : acc {acc*100:5.1f}%   majority {maj*100:5.1f}%")
    print(f"mean conf    : {statistics.mean(local_conf):.3f}")
    a_unc = auc_uncertainty(local_conf, [p == y for p, y in zip(local_pred, ys)])
    print(f"AUC(conf -> correct) {a_unc:.3f}   "
          f"(0.5 = confidence useless, escalation cannot help)")

    # RANDOM-ESCALATION NULL. arXiv 2605.06350 derives that the accuracy gain of
    # score-based escalation over RANDOM escalation decomposes into AUROC terms,
    # so random escalation is the correct null: if our confidence picks rows no
    # better than a coin flip, the whole design is decoration. Any reported gain
    # must be stated relative to this, not relative to tier 1 alone.
    rng = np.random.default_rng(0)
    random_null = []
    for rate in (0.05, 0.1, 0.2, 0.35):
        k = int(round(rate * len(ys)))
        idx = set(rng.choice(len(ys), size=k, replace=False).tolist())
        correct_r, tok_r = 0, 0
        for i in idx:
            r = route(rows[i]["x"], local=None, force_escalate=True,
                      effort=a.effort)
            correct_r += int(r["label"] == ys[i])
            tok_r += r["frontier"]["usage"]["total_tokens"]
        acc_r = (correct_r + sum(1 for i in range(len(ys))
                                if i not in idx and local_pred[i] == ys[i])) / len(ys)
        random_null.append({"rate": rate, "escalated": k, "acc": acc_r,
                            "tokens": tok_r,
                            "tokens_per_100": tok_r * 100 / len(ys)})
        print(f"  RANDOM {rate*100:4.1f}% escalation -> final acc {acc_r*100:5.1f}%  "
              f"tok/100 {tok_r*100/len(ys):7.0f}", flush=True)

    curve = []
    for th in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01):
        need = [i for i, c in enumerate(local_conf) if c < th]
        if a.limit:
            need = need[:a.limit]
        esc_acc, tok = [], 0
        for i in need:
            r = route(rows[i]["x"], local=None, force_escalate=True,
                      effort=a.effort)
            ok = r["label"] == ys[i]
            esc_acc.append(ok)
            tok += r["frontier"]["usage"]["total_tokens"]
        fixed = dict(zip(need, esc_acc))
        correct = sum(fixed.get(i, local_pred[i] == ys[i]) for i in range(len(ys)))
        final = correct / len(ys)
        curve.append({
            "threshold": th, "escalated": len(need),
            "rate": len(need) / len(ys), "acc": final,
            "frontier_acc": (sum(esc_acc) / len(esc_acc)) if esc_acc else None,
            "tokens": tok,
            "tokens_per_100": tok * 100 / len(ys),
        })
        print(f"  th {th:4.2f}  escalate {len(need):4d} "
              f"({len(need)/len(ys)*100:5.1f}%)  final acc {final*100:5.1f}%  "
              f"frontier-acc {(sum(esc_acc)/len(esc_acc)*100 if esc_acc else 0):5.1f}%  "
              f"tok/100 {tok*100/len(ys):7.0f}", flush=True)

    out = {"split": a.split, "n": len(ys), "local_acc": acc, "majority": maj,
           "mean_conf": statistics.mean(local_conf), "auc_conf_correct": a_unc,
           "curve": curve, "random_null": random_null}
    p = os.path.join(ROOT, "data", "responses", f"router_{a.split}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")

    # the honest summary
    best = max(c for c in curve if c["rate"] <= 0.35) if any(
        c["rate"] <= 0.35 for c in curve) else None
    if best:
        print(f"\nat <=35% escalation: final acc {best['acc']*100:.1f}% "
              f"(th {best['threshold']:.2f}, frontier acc "
              f"{(best['frontier_acc'] or 0)*100:.1f}%)")
    # the only comparison that isolates the value of the confidence signal
    for rn in random_null:
        near = min(curve, key=lambda c: abs(c["rate"] - rn["rate"]))
        gain = (near["acc"] - rn["acc"]) * 100
        print(f"  at ~{rn['rate']*100:4.1f}% escalation: score-based {near['acc']*100:5.1f}% "
              f"vs random {rn['acc']*100:5.1f}%  -> gain {gain:+5.1f}pt")

    print("\nNOTE: tier-1 accuracy is reported against a TF-IDF baseline of 92.2%. "
          "The router's value is the curve versus the RANDOM null, not the local number.")


if __name__ == "__main__":
    main()
