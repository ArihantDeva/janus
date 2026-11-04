"""Aggregate per-seed results into a mean and a spread, and refuse to fake it.

WHY THIS EXISTS
The criterion is "three seeds and a mean+spread rather than a single run". A single
number from a single seed is the failure mode this project already hit twice: the
batch-shuffling bug (commit 0715cd1) made three seeds measure init variance only, and
fp16 overflow destroyed one seed while the others trained, so any single-seed report
would have been a clean number over a corrupt model. Aggregation has to be mechanical
so it cannot be quietly skipped.

WHAT IS REPORTED
  mean, stdev, min, max   across seeds, for every metric
  n                       how many seeds actually contributed
  spread                  max - min, which is the honest "how much does this move"
  floor / null            printed once, because they are properties of the data split
                          and must not be averaged as if they were model draws

A seed whose run is marked incomplete, or whose JSON records non-finite training steps,
is EXCLUDED and named. Averaging in a run that diverged is how a spread gets smaller
than the reality.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys


def load_results(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        r["_path"] = os.path.basename(p)
        out.append(r)
    return out


def usable(r):
    """A result is usable only if it finished and never went non-finite.

    A diverged seed is not an outlier to be smoothed away; it is evidence the pipeline
    is numerically fragile on that configuration, and including it would make the
    spread look tighter than the truth.
    """
    reasons = []
    if not r.get("complete", False):
        reasons.append("incomplete")
    for stage, m in (r.get("results") or {}).items():
        if isinstance(m, dict) and m.get("nonfinite_steps", 0) > 0:
            reasons.append(f"{stage} had {m['nonfinite_steps']} non-finite steps")
    return (not reasons), reasons


def summarize(values):
    if not values:
        return None
    return {"n": len(values), "mean": st.mean(values),
            "stdev": st.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
            "spread": max(values) - min(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out_distill"))
    ap.add_argument("--pattern", default="result_seed*.json")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rs = load_results(os.path.join(a.dir, a.pattern))
    if not rs:
        raise SystemExit(f"no result files matched {a.dir}/{a.pattern}")

    good, excluded = [], []
    for r in rs:
        ok, why = usable(r)
        (good if ok else excluded).append((r, why))

    print(f"{len(rs)} result file(s); {len(good)} usable, {len(excluded)} excluded")
    for r, why in excluded:
        print(f"  EXCLUDED {r['_path']}: {'; '.join(why)}")
    if not good:
        raise SystemExit("no usable results -- refusing to report a mean over nothing")

    # `good` holds (result, reasons) pairs so the exclusion reasons stay attached;
    # unpack once here rather than at every use site, which is where the tuple leaked
    # into a `.get` call in the first draft.
    results = [r for r, _why in good]
    floor = [r["tool_only_floor"] for r in results if "tool_only_floor" in r]
    kw = [r["keyword_null"] for r in results if "keyword_null" in r]
    print(f"\ncorpus properties (not averaged as model draws):")
    if floor:
        print(f"  tool-only floor  {100*st.mean(floor):.1f}%")
    if kw:
        print(f"  keyword null     {100*st.mean(kw):.1f}%")

    # per-stage accuracy, ECE, and the post-calibration numbers
    stages = set()
    for r in results:
        stages.update((r.get("results") or {}).keys())
    stages.discard("ood")
    stages.discard("calibration")

    print(f"\n=== across {len(results)} seeds ===")
    agg = {}
    for stage in sorted(stages):
        acc = [r["results"][stage]["acc"] for r in results
               if stage in (r.get("results") or {})]
        ece = [r["results"][stage]["ece"] for r in results
               if stage in (r.get("results") or {})]
        if not acc:
            continue
        s_acc, s_ece = summarize(acc), summarize(ece)
        agg[stage] = {"acc": s_acc, "ece": s_ece}
        print(f"\n  [{stage}]")
        print(f"    acc  {100*s_acc['mean']:5.1f}%  +/- {100*s_acc['stdev']:4.1f}  "
              f"(min {100*s_acc['min']:.1f} max {100*s_acc['max']:.1f}, n={s_acc['n']})")
        print(f"    ece  {s_ece['mean']:.4f}  +/- {s_ece['stdev']:.4f}  "
              f"(min {s_ece['min']:.4f} max {s_ece['max']:.4f})")

    cal_ece = [r["results"]["calibration"]["ece_after"] for r in results
               if "calibration" in (r.get("results") or {})]
    cal_before = [r["results"]["calibration"]["ece_before"] for r in results
                  if "calibration" in (r.get("results") or {})]
    temps = [r["results"]["calibration"]["T"] for r in results
             if "calibration" in (r.get("results") or {})]
    if cal_ece:
        s = summarize(cal_ece)
        sb = summarize(cal_before)
        stt = summarize(temps)
        agg["calibration"] = {"ece_before": sb, "ece_after": s, "T": stt}
        print(f"\n  [calibration, T fitted on the disjoint calib split]")
        print(f"    T    {stt['mean']:.3f} +/- {stt['stdev']:.3f}")
        print(f"    ece  {sb['mean']:.4f} -> {s['mean']:.4f} "
              f"(+/- {s['stdev']:.4f}), max {s['max']:.4f}")
        verdict = "PASS" if s["max"] <= 0.05 else "MISS"
        print(f"    criterion ECE <= 0.05 on EVERY seed: {verdict}")

    ood = [r["results"]["ood"]["acc"] for r in results
           if isinstance((r.get("results") or {}).get("ood"), dict)]
    if ood:
        s = summarize(ood)
        agg["ood"] = s
        print(f"\n  [OOD, tools unseen in training]")
        print(f"    acc  {100*s['mean']:5.1f}%  +/- {100*s['stdev']:4.1f}  "
              f"(n={s['n']} seeds)")

    if a.out:
        json.dump({"n_usable": len(results), "n_excluded": len(excluded),
                   "excluded": [{"file": r["_path"], "why": w} for r, w in excluded],
                   "tool_only_floor": st.mean(floor) if floor else None,
                   "keyword_null": st.mean(kw) if kw else None,
                   "aggregate": agg}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
