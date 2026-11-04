"""Score a candidate model against REAL Janus decisions.

WHAT THIS MEASURES
------------------
Given the states and typed questions from 42 real `janus_ask` calls, compare a
candidate model's distribution with the one Janus actually returned. Three
numbers, each answering a different question:

  mean_total_variation   how far the candidate's distribution is from Janus's
                         (0 = identical, 1 = disjoint). The primary number.
  agreement              how often the candidate's argmax matches Janus's choice.
  brier_vs_reference     proper scoring rule against Janus's probabilities; lower
                         is better, and unlike TV it is minimised by honesty
                         rather than by confidence.

WHY THIS IS NOT A BENCHMARK
---------------------------
The benchmarks ask "is this unsafe", a task nobody actually asked Janus to do.
These are inputs Janus really was asked, with the answers it really gave. That is
the distribution this project is trying to live in, so agreement here is more
relevant than another tenth of a point on ASB.

WHAT IT CANNOT TELL YOU
-----------------------
Janus is not ground truth. Agreement is agreement with a model, and the open-janus
review already recorded the trap: distillation caps you at the teacher and
inherits its biases. A high score here means "looks like Janus", not "is right".

n is small: 81 decisions, grouped in 42 calls. Decisions inside one call share a
state and are therefore correlated, so every interval here is computed by
BATCH bootstrap over calls, never over decisions. Resampling decisions would
pretend independence that does not exist and produce intervals about half as
wide as the truth.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def total_variation(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.shape != q.shape:
        return float("nan")
    return float(0.5 * np.abs(p - q).sum())


def brier(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.mean((p - q) ** 2))


def batch_bootstrap(values, groups, n_boot=2000, seed=0):
    """95% interval by resampling GROUPS (calls), not individual decisions.

    Decisions in one call share a state and are not independent; resampling them
    individually would understate the interval. This is the same lesson as the
    run-29 tie artefact: check the dependence structure before trusting a width.
    """
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        means.append(values[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def score_predictions(decisions, predict):
    """`predict(decision) -> (probs, labels)` for one decision.

    A model that cannot express this decision's label set returns None and is
    counted as `unsupported` rather than scored as wrong -- otherwise a 3-option
    model would be punished for a 4-option Janus question, which inflates the gap
    for a reason that has nothing to do with quality.
    """
    tvs, agrs, briers, groups = [], [], [], []
    unsupported = 0
    for i, d in enumerate(decisions):
        if d["type"] not in ("choice", "noul"):
            continue
        ref = np.array(d["probs"], float)
        got = predict(d)
        if got is None:
            unsupported += 1
            continue
        probs, labels = got
        probs = np.array(probs, float)
        if len(probs) != len(ref):
            unsupported += 1
            continue
        tvs.append(total_variation(probs, ref))
        briers.append(brier(probs, ref))
        agrs.append(float(np.argmax(probs) == np.argmax(ref)))
        groups.append(d.get("ts", "?"))
    if not tvs:
        return {"n": 0, "unsupported": unsupported}
    tv = np.array(tvs)
    return {
        "n": len(tvs),
        "unsupported": unsupported,
        "mean_tv": float(tv.mean()),
        "mean_tv_ci": batch_bootstrap(tv, groups),
        "agreement": float(np.mean(agrs)),
        "brier": float(np.mean(briers)),
        "groups": len(set(groups)),
    }


def report(name, s):
    if not s.get("n"):
        print(f"{name}: no supported decisions ({s.get('unsupported')} unsupported)")
        return
    lo, hi = s["mean_tv_ci"]
    print(f"{name}: n={s['n']} in {s['groups']} calls  "
          f"TV {s['mean_tv']:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"agreement {s['agreement']*100:.1f}%  brier {s['brier']:.4f}  "
          f"unsupported {s['unsupported']}")


def baseline_uniform(decisions):
    """Uniform over each question's own label set. The Brier floor."""
    def predict(d):
        k = len(d["probs"])
        return np.ones(k) / k, d["labels"]
    return predict


def baseline_first(decisions):
    """Always answer the first-listed option with certainty.

    THIS is the honest null, not uniform. Measured on this data: Janus's rendered
    result lists its chosen option FIRST on 44 of 44 choice questions (the
    renderer sorts by probability), so "pick the first one" scores 100%
    agreement on choice questions without doing any work.

    A candidate therefore has to beat this, and any agreement number above it on
    choice questions is meaningless. Recorded here rather than left implicit,
    because it is exactly the kind of position artefact that produced a retracted
    length-control result earlier in this project.
    """
    def predict(d):
        k = len(d["probs"])
        p = np.zeros(k)
        p[0] = 1.0
        return p, d["labels"]
    return predict


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(root, "data", "janus_live", "decisions.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} missing. Run src/janus_live.py first (it reads session logs).")
    decisions = json.load(open(path))
    print(f"real Janus decisions: {len(decisions)}")
    print()
    report("uniform baseline", score_predictions(decisions, baseline_uniform(decisions)))
    report("first-option null", score_predictions(decisions, baseline_first(decisions)))
    print()
    print("The first-option null is the one to beat: Janus's renderer lists the")
    print("chosen option first, so on choice questions it is near-perfect by")
    print("construction. Agreement above it proves nothing on its own.")
    print()
    print("n is small and grouped by call. Intervals are batch bootstraps over")
    print("calls, so they will be wider than a naive per-decision interval.")


if __name__ == "__main__":
    main()
