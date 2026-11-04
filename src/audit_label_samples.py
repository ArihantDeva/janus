"""Are the labels right? Sample each class and check by eye.

Triggered by the audit's own output: the first `reauth` row is pytest progress dots
and the first `abort` row is a pytest ERRORS header. If whole classes are poisoned
the dataset is unusable and no amount of training fixes it.
"""
import json, collections, sys, random
sys.path.insert(0,"/Users/arihantdeva/jevlab/src")
from responses import classify_response
CL=("retry","reauth","fix_call","fix_input","debug","abort")
tr=[r for r in (json.loads(l) for l in open("data/responses/train.jsonl")) if r["y"] in CL]
random.seed(1)
for cls in CL:
    rs=[r for r in tr if r["y"]==cls]
    print(f"\n===== {cls} (n={len(rs)}) =====")
    for r in random.sample(rs, min(5,len(rs))):
        resp, rule = classify_response(r["x"])
        agree = "ok " if resp==cls else "XX "
        print(f"  {agree}rule={str(rule):12s} {r['x'][:105].replace(chr(10),' | ')!r}")
    # how often does the CURRENT rule still agree with the stored label?
    agree = sum(1 for r in rs if classify_response(r["x"])[0]==cls)
    print(f"  stored label agrees with current rules: {agree}/{len(rs)} = {100*agree/len(rs):.1f}%")
