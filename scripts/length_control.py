"""The length control, with the sign fixed A PRIORI.

Run 24 compared the adapter to a length-only baseline and reported that
character count "essentially ties" it. That comparison chose its direction with
max(auc(L), auc(-L)) evaluated ON the benchmark labels -- i.e. it fitted the sign
to the labels it was scoring against. That is the same class of error as every
bug this session found, and it made the baseline look far stronger than an
honest one is.

The correct control picks the direction from the project's OWN dev split and
then applies it unchanged. This script does that and reports both, so the
distinction stays visible:

  a-priori length  : sign from our dev split. THIS is the bar a model must beat.
  label-optimal    : sign fitted on the benchmark. This measures how exploitable
                     the BENCHMARK is by a one-feature rule, not the model.

No GPU and no fitting beyond that single sign, so it is cheap enough to stand as
a permanent control on any future result.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import metrics  # noqa: E402
import evals    # noqa: E402

DATA = os.environ.get("JANUS_DATA", "data")


def best_acc(p, y):
    return max(float((((p >= t).astype(int)) == y).mean()) for t in np.unique(p))


def main():
    dv = [json.loads(l) for l in open(os.path.join(DATA, "safety_dev.jsonl"))]
    dv = [r for r in dv if "exactly one word" in r["prompt"]]
    Ld = np.array([len(r["prompt"]) for r in dv], float)
    yd = np.array([r["label"] for r in dv], float)
    a_dev = metrics.auc(Ld, yd)
    sign = 1 if a_dev > 0.5 else -1
    print(f"our dev: signed length auc {a_dev:.3f} -> a-priori rule is "
          f"score = {sign:+d} * length\n")

    print(f"{'bench':8s} {'floor':>7s} | {'A-PRIORI length':>22s} | "
          f"{'LABEL-OPTIMAL length':>22s}")
    print(f"{'':8s} {'':>7s} | {'auc':>6s} {'acc':>7s} {'lift':>7s} | "
          f"{'auc':>6s} {'acc':>7s} {'lift':>7s}")
    for b in ("rjudge", "asb"):
        rows = evals.load_bench(b)
        L = np.array([len(r["prompt"]) for r in rows], float)
        y = np.array([r["label"] for r in rows], float)
        fl = float(max(y.mean(), 1 - y.mean()))

        p_fixed = sign * L
        a_fixed = metrics.auc(p_fixed, y)
        acc_fixed = best_acc(p_fixed, y)

        # label-fitted sign -- reported ONLY to show benchmark exploitability
        p_fit = L if metrics.auc(L, y) >= metrics.auc(-L, y) else -L
        a_fit = metrics.auc(p_fit, y)
        acc_fit = best_acc(p_fit, y)

        print(f"{b:8s} {fl*100:6.1f}% | {a_fixed:6.3f} {acc_fixed*100:6.1f}% "
              f"{(acc_fixed-fl)*100:+6.1f}pt | {a_fit:6.3f} {acc_fit*100:6.1f}% "
              f"{(acc_fit-fl)*100:+6.1f}pt")
    print("\nA-priori length is the bar a model must beat. The label-optimal "
          "column\nmeasures the benchmark's exploitability, not the model's skill.")
    print("LENGTHCONTROL_DONE")


if __name__ == "__main__":
    main()
