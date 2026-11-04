"""Audit the response dataset. Ask the questions that would discredit it.

Not a summary of what the builder intended -- a check of what is actually in the
files. Six failure modes this project has already hit elsewhere, so each is tested:

  1. LABEL CONTRADICTION  identical text carrying two different labels. The rule is
     ordered so this cannot happen in principle; verify rather than trust.
  2. SPLIT LEAKAGE        the same failure in train and held. Splits are by session
     group, but group assignment is a function of a file path -- if that is wrong,
     leakage is invisible in accuracy and fatal to it.
  3. NEAR-DUPLICATES      whitespace/normalisation variants that survive exact dedup.
  4. TRUNCATION           rows cut at 1500 chars before training sees them, and rows
     that will be cut again by maxlen=768.
  5. LOW INFORMATION      rows that are only boilerplate ("Command exited with code 1").
     A class where these dominate cannot be learned, only memorised.
  6. LABEL PROVENANCE     how many came from the rule vs the unmatched bucket, and
     which rule. A class that is 100% one rule is that rule, not a concept.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import CLASSES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ("train", "rule_held", "unmatched_held")


def load(name):
    p = os.path.join(ROOT, "data", "responses", f"{name}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def norm(t):
    """Aggressive normalisation used ONLY for duplicate detection."""
    t = t.lower()
    t = re.sub(r"/[\w./~-]{4,}", "<path>", t)     # collapse paths
    t = re.sub(r"\b\d+(\.\d+)?\b", "N", t)        # collapse numbers
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    data = {s: load(s) for s in SPLITS}
    allrows = [(s, r) for s in SPLITS for r in data[s]]

    print("=" * 78)
    print("1. WHAT IT IS")
    print("=" * 78)
    for s in SPLITS:
        rs = data[s]
        if not rs:
            print(f"  {s:16s} MISSING")
            continue
        cl = collections.Counter(r["y"] for r in rs if r["y"] in CLASSES)
        other = sum(1 for r in rs if r["y"] not in CLASSES)
        print(f"  {s:16s} {len(rs):6d} rows, {len(cl)} classes in CLASSES, "
              f"{other} outside")
        for k, v in cl.most_common():
            print(f"      {k:11s} {v:5d}  {100*v/len(rs):5.1f}%")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. LABEL CONTRADICTION: identical text, different labels")
    print("=" * 78)
    by_text = collections.defaultdict(set)
    for s, r in allrows:
        by_text[tuple(norm(r["x"]))].add(r["y"])
    contra = {k: v for k, v in by_text.items() if len(v) > 1}
    print(f"  distinct normalised texts: {len(by_text)}")
    print(f"  texts with >1 label      : {len(contra)}")
    for k, v in list(contra.items())[:5]:
        print(f"    labels={sorted(v)}  text={k[:110]}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. SPLIT LEAKAGE: same normalised text across splits")
    print("=" * 78)
    where = collections.defaultdict(set)
    for s, r in allrows:
        where[tuple(norm(r["x"]))].add(s)
    multi = {k: v for k, v in where.items() if len(v) > 1}
    print(f"  texts appearing in >1 split: {len(multi)} "
          f"({100*len(multi)/max(1,len(where)):.1f}% of distinct texts)")
    cross_th = [k for k, v in multi.items() if {"train", "rule_held"} <= v]
    print(f"  ...appearing in BOTH train and rule_held: {len(cross_th)}")
    for k in cross_th[:4]:
        print(f"    {k[:120]}")

    # group ids must not overlap
    g = {s: set(r["group"] for r in data[s]) for s in SPLITS}
    print(f"\n  group ids per split: " +
          " ".join(f"{s}={len(g[s])}" for s in SPLITS))
    for a in SPLITS:
        for b in SPLITS:
            if a < b and g[a] and g[b]:
                ov = g[a] & g[b]
                print(f"  overlap {a}/{b}: {len(ov)}"
                      + ("  <-- LEAK" if ov else "  ok"))

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. EXACT DUPLICATES WITHIN A SPLIT")
    print("=" * 78)
    for s in SPLITS:
        c = collections.Counter(r["x"].strip() for r in data[s])
        dup = sum(v - 1 for v in c.values() if v > 1)
        print(f"  {s:16s} {len(data[s]):6d} rows, {len(c):6d} unique, "
              f"{dup:5d} redundant copies ({100*dup/max(1,len(data[s])):.1f}%)")
        top = c.most_common(2)
        for t, n in top:
            if n > 1:
                print(f"      x{n:4d}  {t[:90]!r}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. TRUNCATION")
    print("=" * 78)
    for s in SPLITS:
        rs = data[s]
        if not rs:
            continue
        n1500 = sum(1 for r in rs if len(r["x"]) >= 1500)
        est = sum(1 for r in rs if (len(r["x"]) / 3.5 + 600) > 768)
        print(f"  {s:16s} at 1500-char cap: {n1500:5d} ({100*n1500/len(rs):4.1f}%)"
              f"   will exceed maxlen 768: {est:5d} ({100*est/len(rs):4.1f}%)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. LOW-INFORMATION ROWS (boilerplate only)")
    print("=" * 78)
    boiler = re.compile(
        r"^\W*(command (exited|failed)[^.]*|exit (code|status) \d+|"
        r"\(no output\)|no output|failed|error|unknown error)\W*$", re.I)
    for s in ("train", "rule_held"):
        rs = [r for r in data[s] if r["y"] in CLASSES]
        low = [r for r in rs if boiler.match(r["x"].strip()) or len(r["x"].strip()) < 12]
        print(f"  {s:16s} {len(low):5d} / {len(rs)} ({100*len(low)/max(1,len(rs)):.1f}%)")
        if low:
            c = collections.Counter(r["y"] for r in low)
            print(f"      by class: {dict(c.most_common())}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("7. LABEL PROVENANCE: which rule produced each label")
    print("=" * 78)
    tr = [r for r in data["train"] if r["y"] in CLASSES]
    rules = collections.Counter(r.get("rule") for r in tr)
    print(f"  train rows with a rule      : {sum(v for k,v in rules.items() if k)}")
    print(f"  train rows with rule=None   : {rules.get(None, 0)}")
    print("\n  rule -> class mapping (a class produced by ONE rule is that rule):")
    pair = collections.Counter((r.get("rule"), r["y"]) for r in tr)
    per_class = collections.Counter(r["y"] for r in tr)
    for (rule, cls), n in pair.most_common(24):
        share = 100 * n / per_class[cls]
        flag = "   <-- single-rule" if share > 95 else ""
        print(f"    {str(rule):14s} -> {cls:11s} {n:5d}  {share:5.1f}% of class{flag}")

    print("\n" + "=" * 78)
    print("8. WHAT A JUDGE WOULD SEE FIRST")
    print("=" * 78)
    for cls in CLASSES:
        rs = [r for r in data["train"] if r["y"] == cls]
        if rs:
            print(f"  {cls:11s} n={len(rs):5d}  e.g. {rs[0]['x'][:90].strip()!r}")


if __name__ == "__main__":
    main()
