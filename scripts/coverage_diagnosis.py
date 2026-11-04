"""Why is cov@5 always zero? Answer it instead of assuming a metric bug.

cov@5 has been 0 or 0.01 in every run of this project, zero-shot and finetuned,
on both benchmarks. A number that never varies across 25 runs and ten fixed bugs
looks like a broken metric, so this checks the metric first and then explains it.

Findings this produces (verified against two independent definitions of
coverage, which agree):

  - the implementation is correct;
  - confidence tracks WHICH CLASS was predicted, not whether the prediction is
    right (mean confidence 0.74-0.76 when predicting unsafe vs 0.63-0.64 when
    predicting safe);
  - within a predicted class, confidence ranks correctness at auc barely above
    chance;
  - therefore the best-ranked subset is already ~32% wrong and a 5% error budget
    is unreachable by ANY monotone recalibration.

The last point is the one that matters: coverage at a small budget cannot be
created by temperature, thresholds or calibration, because all of them are
monotone in the score and cannot reorder items. Coverage is a property of the
RANKING, and this ranking is flat.

No GPU needed; it reads saved probabilities or recomputes nothing expensive.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import metrics  # noqa: E402


def coverage_accepted_set(p, y, budget=0.05):
    """Textbook selective prediction: risk measured on the accepted set."""
    conf = np.maximum(p, 1 - p)
    wrong = (1.0 - metrics._correct(p, y)).astype(int)
    order = np.argsort(-conf)
    w = wrong[order]
    best = 0
    for k in range(1, len(w) + 1):
        if w[:k].mean() <= budget:
            best = k
        else:
            break
    return best / len(w), best


def report(npz, benches=("rjudge", "asb"), seeds=3):
    d = np.load(npz)
    print("=== metric check: two independent coverage definitions ===")
    agree = True
    for b in benches:
        y = d[f"{b}_s0_y"]
        for i in range(seeds):
            p = d[f"{b}_s{i}_p"]
            a = metrics.coverage_at(p, y, 0.05)[0]
            c = coverage_accepted_set(p, y, 0.05)[0]
            agree &= abs(a - c) < 1e-9
    print(f"  definitions agree on every seed: {agree}")

    print("\n=== confidence by predicted class ===")
    for b in benches:
        y = d[f"{b}_s0_y"].astype(int)
        for i in range(seeds):
            p = d[f"{b}_s{i}_p"]
            pred = (p >= 0.5).astype(int)
            conf = np.maximum(p, 1 - p)
            cu = conf[pred == 1].mean()
            cs = conf[pred == 0].mean() if (pred == 0).any() else float("nan")
            print(f"  {b} seed{i}: unsafe-pred conf {cu:.3f} (n={sum(pred==1):4d})  "
                  f"safe-pred conf {cs:.3f} (n={sum(pred==0):4d})")

    print("\n=== can confidence rank CORRECTNESS within a predicted class? ===")
    for b in benches:
        y = d[f"{b}_s0_y"].astype(int)
        for i in range(seeds):
            p = d[f"{b}_s{i}_p"]
            pred = (p >= 0.5).astype(int)
            correct = metrics._correct(p, y).astype(int)
            conf = np.maximum(p, 1 - p)
            cells = []
            for cls in (1, 0):
                m = pred == cls
                if m.sum() < 20 or len(np.unique(correct[m])) < 2:
                    cells.append("n/a"); continue
                cells.append(f"auc {metrics.auc(conf[m], correct[m]):.3f}")
            print(f"  {b} seed{i}: unsafe-pred {cells[0]}  safe-pred {cells[1]}")

    print("\n=== reachable operating points (coverage, mean over seeds) ===")
    for b in benches:
        y = d[f"{b}_s0_y"]
        for bud in (0.05, 0.10, 0.20, 0.30, 0.40):
            cs = [metrics.coverage_at(d[f"{b}_s{i}_p"], y, bud)[0] * 100
                  for i in range(seeds)]
            print(f"  {b:8s} budget {bud*100:3.0f}%: coverage {np.mean(cs):5.1f}%")
    print("\nCOVERAGE_DONE")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "/scratch/ad2416/loopout/oracle.npz")
