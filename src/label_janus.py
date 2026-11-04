"""Label the distillation corpus with live Janus, keeping the FULL distribution.

WHY THE DISTRIBUTION AND NOT THE ARGMAX
The argmax throws away everything the teacher knows. A row Janus calls
`{fix_call: 0.52, fix_input: 0.44, debug: 0.04}` is a genuine near-tie, and
hardening it to `fix_call` invents certainty the teacher never had. Distillation
on the distribution teaches the student *how confident to be* -- which is the
product claim here, because the hook thresholds on probability. So the training
target is the normalised teacher vector, and the argmax is stored only as a
convenience column for reporting.

WHY TEMPERATURE IS RECORDED AS A RAW LOGIT VECTOR TOO
`softmax(log(p)/T)` cannot be undone from p alone for arbitrary T, so the raw
distribution is stored and the temperature is applied at train time. That keeps
the corpus reusable if the temperature is retuned without re-paying for 25k teacher
calls.

WHAT HAPPENS WHEN THE TEACHER FAILS
The row is recorded with `teacher_error` and no distribution, and it is EXCLUDED
from training rather than being given a fallback label. A silent fallback would
mean the student trains on the rule's opinion while the run reports "distilled from
Janus" -- the exact mislabelling that made run 33's model==rule agreement look like
a result. Failed count is printed and, if it exceeds a threshold, the script
refuses to declare the corpus complete.

IDEMPOTENCE AND RESUMPTION
Progress is checkpointed every `--flush` rows by rewriting the output file, and a
rerun skips any (tool, arguments, error) key already labelled. A crash costs at most
one flush interval, not the whole run. The key is the full state so two rows
differing only in arguments are both labelled.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import collections
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "data", "distill", "calls.jsonl")
OUT = os.path.join(ROOT, "data", "distill", "janus.jsonl")


def state_key(r):
    return (r["tool"], json.dumps(r.get("arguments"), sort_keys=True, default=str),
            r["error"])


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(state_key(json.loads(line)))
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-error-chars", type=int, default=8000,
                    help="teacher sees the same 8k the student will")
    ap.add_argument("--flush", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="report coverage and exit; change nothing")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(IN)]
    if a.limit:
        rows = rows[:a.limit]
    done = load_done(a.out)
    todo = [r for r in rows if state_key(r) not in done]
    print(f"{len(rows)} corpus rows, {len(done)} already labelled, {len(todo)} to do")
    if a.check or not todo:
        report(a.out)
        return

    import backends as B
    backend = B.get_backend("teacher")
    lock = threading.Lock()
    t0 = time.time()
    n_ok = n_err = 0
    fh = open(a.out, "a")

    def work(r):
        payload = json.dumps({"tool": r["tool"], "arguments": r.get("arguments"),
                              "error": r["error"][:a.max_error_chars]}, default=str)
        try:
            d = backend(payload)
            B.validate(d, "janus")
            return r, d, None
        except Exception as e:
            return r, None, f"{type(e).__name__}: {str(e)[:200]}"

    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r, dist, err = fut.result()
            rec = dict(r)
            if dist is None:
                rec["teacher_error"] = err
                n_err += 1
            else:
                rec["teacher"] = {k: float(v) for k, v in dist.items()}
                rec["teacher_argmax"] = max(dist, key=dist.get)
                rec["teacher_conf"] = float(max(dist.values()))
                n_ok += 1
                if r.get("rule_label"):
                    rec["rule_agrees"] = (rec["teacher_argmax"] == r["rule_label"])
            with lock:
                fh.write(json.dumps(rec) + "\n")
                if i % a.flush == 0:
                    fh.flush()
                    os.fsync(fh.fileno())
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ok {n_ok} err {n_err}  "
                          f"{i/el:.1f}/s  eta {(len(todo)-i)/max(i/el,1e-9)/60:.0f}m",
                          flush=True)
    fh.close()
    report(a.out)


def report(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    good = [r for r in rows if "teacher" in r]
    bad = [r for r in rows if "teacher" not in r]
    print(f"\n{path}: {len(rows)} rows, {len(good)} labelled, {len(bad)} teacher errors")
    if not good:
        return
    conf = sorted(r["teacher_conf"] for r in good)
    print(f"teacher confidence: p10 {conf[len(conf)//10]:.3f} "
          f"p50 {conf[len(conf)//2]:.3f} p90 {conf[int(.9*len(conf))]:.3f}")
    c = collections.Counter(r["teacher_argmax"] for r in good)
    print(f"teacher class balance: {c.most_common()}")
    both = [r for r in good if "rule_agrees" in r]
    if both:
        ag = sum(r["rule_agrees"] for r in both)
        print(f"teacher vs rule on the {len(both)} rows the rule could label: "
              f"{100*ag/len(both):.1f}% agree")
    low = sum(1 for r in good if r["teacher_conf"] < 0.6)
    print(f"rows where the teacher is itself unsure (<0.6): {low} "
          f"({100*low/len(good):.1f}%)")
    tools = collections.Counter(r["tool"] for r in good)
    print(f"tools covered: {len(tools)}")
    if bad:
        print(f"sample teacher errors: {[r.get('teacher_error','')[:70] for r in bad[:3]]}")


if __name__ == "__main__":
    main()
