"""Final numbers. Everything reported is on data the rules never saw.

unm_dev  : used to DISCOVER the missing classes (unread by any rule design before)
unm_held : NEVER touched during rule design. The only numbers that count.
"""
import json, collections, sys
import numpy as np
sys.path.insert(0, "/Users/arihantdeva/jevlab/src")
from responses import classify_response, UNMATCHED
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def load(n): return [json.loads(l) for l in open(f"data/responses/{n}.jsonl")]

tr = [r for r in load("train")]
held = load("rule_held")
unm_held = load("unm_held")

Xtr=[r["x"] for r in tr]; ytr=[r["y"] for r in tr]
print(f"train {len(tr)}   rule_held {len(held)}   unm_held {len(unm_held)}")
print()

# constant baseline on rule_held
top = collections.Counter(ytr).most_common(1)[0][0]
print(f"rule_held  majority({top})        {100*np.mean([r['y']==top for r in held]):5.1f}%")

clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=40000, sublinear_tf=True),
    LogisticRegression(max_iter=400, class_weight="balanced"))
clf.fit(Xtr, ytr)

Xhe=[r["x"] for r in held]; yhe=np.array([r["y"] for r in held])
pred=clf.predict(Xhe); conf=clf.predict_proba(Xhe).max(axis=1)
acc=(pred==yhe).mean()
agree=(pred==np.array([classify_response(x)[0] for x in Xhe])).mean()
print(f"rule_held  lexical model        {100*acc:5.1f}%")
print(f"rule_held  model==rule          {100*agree:5.1f}%   <- how much it IS the regex")
print(f"rule_held  mean confidence      {conf.mean():.2f}")
print()
print("per-class recall on rule_held:")
for k in sorted(set(yhe)):
    m=yhe==k
    print(f"  {k:12s} n={m.sum():5d}  {100*np.mean(pred[m]==k):5.1f}%")

# --- the honest generalisation test ---
print()
print("=== unm_held: rows SHIPPED with no rule match ===")
Xu=[r["x"] for r in unm_held]
pu=clf.predict(Xu); cu=clf.predict_proba(Xu).max(axis=1)
print(f"  n={len(Xu)}  mean confidence {cu.mean():.2f}  confident(>0.6) {100*np.mean(cu>0.6):.1f}%")
print(f"  predicted: {dict(collections.Counter(pu).most_common())}")
print()
print("  A model that generalised would be CONFIDENT here. Measured confidence")
print("  is the evidence, not accuracy -- there is no ground truth on this set.")
