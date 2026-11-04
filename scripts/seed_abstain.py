"""Does seed disagreement buy a usable abstention signal?

CONTEXT
-------
The guard's operational requirement is not "be right more often". It is "know
when you are not sure". coverage_at() in metrics.py ranks by

    _confidence(p) = max(p, 1-p)

which is the model's OWN distance from 0.5. That number cannot separate
"confidently right" from "confidently wrong", and the consequence is measured:
cov@5 has been structurally 0 across 28 runs on both benchmarks (run 26). No
monotone recalibration can fix it, because rescaling does not reorder the rows.

THE HYPOTHESIS
--------------
Three seeds trained on identical data differ only in initialisation and batch
order. Where they agree, the decision is stable; where they disagree, it is an
artefact of the training run rather than a property of the input. That
disagreement is EPISTEMIC uncertainty, and it is independent of the model's own
confidence -- so ranking by it can reorder rows in a way temperature cannot.

WHY THIS IS NOT RUN 15 AGAIN
----------------------------
Run 15 ensembled the seeds by averaging probabilities and asked whether
ACCURACY improved. It did not, and that result stands: the seeds are not diverse
enough for the mean to beat the best single member. This script never averages a
prediction. It leaves the point prediction exactly alone and only asks whether
the SPREAD tells you which rows to skip. A worse mean is compatible with a
useful spread; they are different questions.

WHAT IS REPORTED
----------------
For each benchmark, coverage at 5% and 10% error budget under three rankings:
  own    max(p,1-p)        -- the current shipped rule (baseline)
  spread -sd(p_i)          -- high when seeds agree
  entropy-like: -(p^3+(1-p)^3) ... kept simple, see below
A ranking is only useful if it puts CORRECT rows first. So the decisive number
is the AUC of each uncertainty score against the correctness indicator.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/scratch/ad2416/janussrc")


def _correct(p, y):
    return ((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).astype(float)


def cov_at(conf, correct, budget=0.05):
    """Identical semantics to metrics.coverage_at: rank by descending conf,
    require EVERY prefix to stay within budget."""
    conf = np.asarray(conf, float)
    wrong = 1.0 - np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    w = wrong[order].astype(int)
    n = len(w)
    running = np.cumsum(w) / np.arange(1, n + 1)
    ok = running <= budget
    # prefix is valid only up to the first violation
    bad = np.argmax(~ok) if (~ok).any() else n
    return bad / n


def auc_unc(unc, correct):
    """AUC of an uncertainty score at predicting WRONGNESS, via rank statistic.
    Higher uncertainty should mean more likely wrong, so we correlate unc with
    (1-correct). Reported as AUC in [0,1]; 0.5 = no information."""
    unc = np.asarray(unc, float)
    target = 1.0 - np.asarray(correct, float)
    if target.all() or (~target.astype(bool)).all():
        return float("nan")
    order = np.argsort(unc, kind="stable")
    ranks = np.empty(len(unc), float)
    ranks[order] = np.arange(1, len(unc) + 1)
    n1 = target.sum()
    n0 = len(target) - n1
    return float((ranks[target.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/scratch/ad2416"
    per_bench = {"rjudge": [], "asb": []}
    for s in (0, 1, 2):
        d = json.load(open(f"{root}/seedrun{s}/bench.json"))
        for entry in d:
            # bench.json is a list, one dict per benchmark, with a "rows" block
            name = "rjudge" if "rjudge" in json.dumps(entry.get("bench", "rjudge")).lower() else "asb"
            per_bench[name].append(entry)

    for bench in ("rjudge", "asb"):
        runs = per_bench[bench]
        if not runs:
            print(f"{bench}: no data")
            continue
        # rows are dumped per seed; align on row count and verify labels agree
        ys = [np.array(r["rows"]["y"]) for r in runs]
        if not all(len(y) == len(ys[0]) for y in ys) or not all((y == ys[0]).all() for y in ys):
            print(f"{bench}: ROW MISMATCH across seeds -- refusing to align")
            continue
        y = ys[0]
        P = np.vstack([np.array(r["rows"]["p_finetuned_T"]) if "p_finetuned_T" in r["rows"]
                       else np.array(r["rows"]["p_finetuned"]) for r in runs])
        mean = P.mean(axis=0)
        spread = P.std(axis=0)

        print(f"\n=== {bench.upper()}  n={len(y)}  seeds={P.shape[0]} ===")
        print(f"  mean pairwise seed sd: {spread.mean():.4f}")

        # single-seed reference
        for i in range(P.shape[0]):
            c = _correct(P[i], y)
            own = np.maximum(P[i], 1 - P[i])
            print(f"  seed{i}: acc {c.mean()*100:5.1f}%  cov@5(own) {cov_at(own, c, .05)*100:4.1f}%"
                  f"  auc {auc_unc(1-own, c):.3f}")

        c_mean = _correct(mean, y)
        own = np.maximum(mean, 1 - mean)
        # uncertainty candidates
        cands = {
            "own(mean)": 1 - own,                 # current rule, high = unsure
            "spread": spread,                      # seed disagreement, high = unsure
            "own+spread": (1 - own) + spread,      # combined
        }
        print(f"  --- mean-prob point prediction: acc {c_mean.mean()*100:5.1f}% ---")
        for nm, unc in cands.items():
            conf = -unc
            print(f"    {nm:12s} cov@5 {cov_at(conf, c_mean, .05)*100:5.1f}%"
                  f"  cov@10 {cov_at(conf, c_mean, .10)*100:5.1f}%"
                  f"  auc_unc {auc_unc(unc, c_mean):.3f}")


if __name__ == "__main__":
    main()
