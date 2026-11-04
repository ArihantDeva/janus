"""Two checks before committing a fine-tune to the fix_call gap.

CHECK 1: is fix_call the class most damaged by truncation?
  maxlen=768 with ~600 tokens of prompt overhead leaves ~170 tokens of failure
  text. Edit failures (fix_call) carry paths + oldText excerpts and are plausibly
  the longest. If so, the fine-tune CANNOT learn fix_call for a mechanical reason,
  and weighting or relabelling would be treating a symptom.

CHECK 2: is the keyword null a faithful ablation of the label rule?
  Each backend has its own ordered rule list. If they order the classes
  differently, the null answers a different question and its 75% is not a fair
  comparison to the real rule.
"""
import json, re, sys, collections
sys.path.insert(0, "/Users/arihantdeva/jevlab/src")
import numpy as np
from responses import RULES, classify_response
from backends import KEYWORDS

CL = ("retry","reauth","fix_call","fix_input","debug","abort")

# ---- CHECK 1: length per class -------------------------------------------
tr = [r for r in (json.loads(l) for l in open("data/responses/train.jsonl")) if r["y"] in CL]
print("=== CHECK 1: length by class (chars, then est. truncation) ===")
print(f"{'class':11s} {'n':>5s} {'p50':>6s} {'p90':>6s} {'p99':>6s} {'max':>6s} {'est_trunc@768':>14s}")
# ~3.5 chars/token; prompt overhead measured ~600 tokens
def trunc_rate(rs):
    return np.mean([ (len(r["x"])/3.5 + 600) > 768 for r in rs ])
rows=[]
for c in CL:
    rs=[r for r in tr if r["y"]==c]
    L=np.array([len(r["x"]) for r in rs])
    rows.append((c,len(rs),np.percentile(L,50),np.percentile(L,90),np.percentile(L,99),L.max(),trunc_rate(rs)))
    print(f"{c:11s} {len(rs):5d} {np.percentile(L,50):6.0f} {np.percentile(L,90):6.0f} {np.percentile(L,99):6.0f} {L.max():6.0f} {100*trunc_rate(rs):13.1f}%")
worst=max(rows,key=lambda r:r[6])
print(f"\n  worst-truncated class: {worst[0]} ({100*worst[6]:.1f}%)")

# ---- CHECK 2: rule order fidelity ----------------------------------------
print("\n=== CHECK 2: class ordering in the two rule lists ===")
# flatten: for each rule, which class does it emit? (responses.RULES is (name, regex) -> name)
print("  responses.RULES order (first match wins):")
for i,(name,_rx) in enumerate(RULES[:14]):
    print(f"    {i:2d} {name}")
print("\n  backends.KEYWORDS order (first match wins):")
for i,(cls,_keys) in enumerate(KEYWORDS):
    print(f"    {i:2d} {cls}")

# does keyword order agree with rule order on where fix_call sits?
rule_pos = {n:i for i,(n,_) in enumerate(RULES)}
kw_pos = {c:i for i,(c,_) in enumerate(KEYWORDS)}
print("\n  position of each class (lower fires earlier):")
for c in CL:
    rp = rule_pos.get(c, rule_pos.get("fix_call" if c=="fix_call" else c, "-"))
    print(f"    {c:11s} rule={str(rp):>3s}  keyword={kw_pos.get(c,'-'):>3}")

# ---- CHECK 3: where keyword disagrees with the rule, what does it say? ----
print("\n=== CHECK 3: on rule-labelled train rows, keyword vs rule ===")
from backends import KeywordBackend
kw = KeywordBackend()
conf = collections.Counter()
for r in tr:
    k = max(kw(r["x"]), key=lambda d: kw(r["x"])[d])
    conf[(r["y"], k)] += 1
print("  confusion (rule_label -> keyword_label), top 12:")
for (a,b),n in conf.most_common(12):
    mark = "  OK" if a==b else "  MISMATCH"
    print(f"    {a:10s} -> {b:10s} {n:5d}{mark}")
hit = sum(n for (a,b),n in conf.items() if a==b)/sum(conf.values())
print(f"\n  keyword agreement with the rule on TRAIN rows: {100*hit:.1f}%")
print("  (this is the null's CEILING; its 75% on held rows is below it only by")
print("   distribution shift, not by any hidden capability)")
