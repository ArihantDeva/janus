"""Can a linear model trained on OUR corpus reach the benchmarks?

Run 20 measured TF-IDF at AUC 0.974 (rjudge) / 0.851 (asb) on the rendered
benchmark prompts -- but with 5-fold cross-validation ON THE BENCHMARK LABELS.
That is a diagnostic about how much signal the text carries, not a result: it
fits on the very labels it is scored against.

This is the honest version. The classifier is trained ONLY on the project's own
corpus (data/safety_*.jsonl, the tool-call guard task) and then evaluated on the
benchmarks, which it has never seen. No benchmark label is used for fitting,
feature selection, threshold choice or regularisation -- the defaults are fixed
here and C is never tuned on a benchmark.

Two questions this answers:
  1. Does the corpus carry enough to transfer? If a linear model trained on our
     data cannot beat the floor either, then the problem is the DATA and no
     representation change will save it.
  2. If it does transfer, the adapter is not merely poorly tuned -- it is losing
     information a bag-of-words keeps, and the lever is representation.

Reported alongside the adapter's own n=3 numbers for the same prompts.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import evals  # noqa: E402

DATA = os.environ.get("JANUS_DATA", "data")


def load_corpus():
    tr, dv = [], []
    for name, acc in (("safety_train.jsonl", tr), ("safety_dev.jsonl", dv)):
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            continue
        for ln in open(p):
            r = json.loads(ln)
            # only rows carrying an answer contract are the same TASK
            if "exactly one word" in r["prompt"]:
                acc.append((r["prompt"], int(r["label"])))
    return tr, dv


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    tr, dv = load_corpus()
    Xtr = [x for x, _ in tr]; ytr = np.array([y for _, y in tr])
    Xdv = [x for x, _ in dv]; ydv = np.array([y for _, y in dv])
    print(f"own corpus: train {len(ytr)} (unsafe {ytr.mean():.3f}), "
          f"dev {len(ydv)} (unsafe {ydv.mean():.3f})")
    print("            (296 SWE-style rows without an answer contract excluded: "
          "different task)")

    # C and ngrams are FIXED here and were not chosen against any benchmark.
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    Ztr = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Ztr, ytr)

    # sanity: the model must work on our own held-out split first
    pdv = clf.predict_proba(vec.transform(Xdv))[:, 1]
    print(f"own dev    : auc {roc_auc_score(ydv, pdv):.3f}  "
          f"acc {((pdv>=0.5).astype(int)==ydv).mean()*100:.1f}%")
    print()

    print(f"{'bench':8s} {'n':>5s} {'floor':>7s} {'tfidf acc':>10s} {'lift':>8s} "
          f"{'auc':>6s}   adapter(n=3): auc / acc")
    adapter = {"rjudge": (0.645, 0.658), "asb": (0.5603, 0.618)}
    for b in ("rjudge", "asb"):
        rows = evals.load_bench(b)
        X = [r["prompt"] for r in rows]; y = np.array([r["label"] for r in rows])
        fl = float(max(y.mean(), 1 - y.mean()))
        p = clf.predict_proba(vec.transform(X))[:, 1]
        acc = float(((p >= 0.5).astype(int) == y).mean())
        auc = float(roc_auc_score(y, p))
        a_auc, a_acc = adapter[b]
        print(f"{b:8s} {len(y):5d} {fl*100:6.1f}% {acc*100:9.1f}% "
              f"{(acc-fl)*100:+7.1f}pt {auc:6.3f}   {a_auc:.3f} / {a_acc*100:.1f}%")
    print("\nLINEAR_DONE")


if __name__ == "__main__":
    main()
