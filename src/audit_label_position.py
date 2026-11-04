"""Where does each label actually come from, and is that a real signal?

Hypothesis: labels are contaminated because the rules match SUBSTRINGS anywhere in
a 1500-char pasted log, so short tokens ("auth", "token", "retry", "abort", "403")
fire on incidental code/log content far from the actual error.

Test: for every labelled row, find WHICH alternative matched and WHERE it matched
(first line = the error header, or deep in the body). A label whose match sits at
char 900 of a pasted log is describing something incidental.
"""
import json, re, sys, collections
sys.path.insert(0,"/Users/arihantdeva/jevlab/src")
from responses import RULES
CL=("retry","reauth","fix_call","fix_input","debug","abort")
tr=[r for r in (json.loads(l) for l in open("data/responses/train.jsonl")) if r["y"] in CL]

# split each rule's pattern into top-level alternatives to see which fired
def alternatives(rx):
    p = rx.pattern
    # crude but adequate: split on unescaped | at depth 0
    out, depth, cur, i = [], 0, "", 0
    while i < len(p):
        c = p[i]
        if c == "\\" and i+1 < len(p):
            cur += p[i:i+2]; i += 2; continue
        if c == "(": depth += 1
        elif c == ")": depth -= 1
        if c == "|" and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += c
        i += 1
    out.append(cur)
    return [a for a in out if a]

print("=== which rule alternative fires, and where ===")
pos_bucket = collections.Counter()
alts_hit = collections.defaultdict(collections.Counter)
for r in tr:
    t = r["x"]; first_nl = t.find("\n")
    header_end = first_nl if first_nl > 0 else len(t)
    for name, rx in RULES:
        m = None
        for a in alternatives(rx):
            try: mm = re.search(a, t, rx.flags)
            except re.error: continue
            if mm: m = (a, mm.start()); break
        if m:
            alt, pos = m
            where = "header" if pos < header_end else ("line2-3" if pos < 300 else "deep")
            pos_bucket[(r["y"], where)] += 1
            if r["y"] in CL:
                alts_hit[r["y"]][alt[:46]] += 1
            break

print(f"\n{'class':11s} {'header':>7s} {'line2-3':>8s} {'deep':>6s}   (deep = label came from mid-log content)")
for cls in CL:
    h=pos_bucket[(cls,"header")]; m=pos_bucket[(cls,"line2-3")]; d=pos_bucket[(cls,"deep")]
    tot=h+m+d
    if tot: print(f"{cls:11s} {h:7d} {m:8d} {d:6d}   deep={100*d/tot:5.1f}%")

print("\n=== top alternatives per class (the actual label source) ===")
for cls in CL:
    print(f"\n  {cls}:")
    for a,n in alts_hit[cls].most_common(6):
        print(f"    {n:5d}  {a}")
