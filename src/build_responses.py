"""Build the response-classification dataset from real tool failures.

INPUT is the error text an agent received. OUTPUT is the correct response class
from responses.py. This is the switch an agent makes dozens of times a session.

THE TRAIN/TEST DESIGN IS THE WHOLE POINT
----------------------------------------
Labels come from a regex over the same text the model reads, so a naive split
would measure "can a model reproduce my regex" -- a tautology. The split is
therefore:

    train   rows the rule labelled confidently (it matched something)
    test    two separate sets:
              rule_held   rows the rule matched, unseen sessions
              unmatched   rows the rule could NOT label

`unmatched` is the real question. Those rows are where the rule failed, and
predicting them requires a notion of what an error MEANS rather than pattern
matching. A model that scores well there has generalised; one that scores well
only on rule_held has learned my regex.

Measured context: 16636 non-guard failures, of which 7377 (44%) match no rule.
That is a large, honest evaluation set the teacher could not label.

SPLIT BY SESSION GROUP: failures repeat across sessions (one text 551 times), so
a row split leaks. Group = project + session id.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from responses import UNMATCHED, classify_response  # noqa: E402
from tool_outcomes import GUARD, _session_group, _text_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORES = (".pi/agent/sessions", ".pi-bolt/agent/sessions")


def harvest(stores=STORES, home="~"):
    rows = []
    files = []
    for s in stores:
        d = os.path.join(os.path.expanduser(home), s)
        if os.path.isdir(d):
            files += glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)
    for f in files:
        group = _session_group(f)
        try:
            fh = open(f, errors="ignore")
        except Exception:
            continue
        for ln in fh:
            if "toolResult" not in ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            m = r.get("message") or {}
            if m.get("role") != "toolResult" or not m.get("isError"):
                continue
            txt = _text_of(m)
            if not txt.strip():
                continue
            if GUARD.search(txt[:400]):
                continue
            label, rule = classify_response(txt)
            rows.append({"x": txt[:1500], "y": label, "rule": rule,
                         "tool": m.get("toolName"), "group": group})
        fh.close()
    return rows


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        k = (r["x"].strip(), r["y"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def split(rows, seed=0, frac=0.25):
    """Group-disjoint split, then partition the test side by rule match."""
    import numpy as np
    rng = np.random.default_rng(seed)
    groups = np.array([r["group"] for r in rows])
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    cut = int(len(uniq) * (1 - frac))
    tr_g, te_g = set(uniq[:cut]), set(uniq[cut:])
    tr = [r for r in rows if r["group"] in tr_g]
    te = [r for r in rows if r["group"] in te_g]
    train = [r for r in tr if r["y"] != UNMATCHED]
    held = [r for r in te if r["y"] != UNMATCHED]
    unm = [r for r in te if r["y"] == UNMATCHED]
    return train, held, unm


def main():
    ap_out = os.path.join(ROOT, "data", "responses")
    os.makedirs(ap_out, exist_ok=True)
    print("harvesting failures...")
    rows = harvest()
    print(f"  raw failures        : {len(rows)}")
    rows = dedup(rows)
    print(f"  after exact dedup   : {len(rows)}")

    train, held, unm = split(rows)
    print(f"\n  train  (rule-labelled, unseen-rule-free) : {len(train)}")
    print(f"  test/rule_held                           : {len(held)}")
    print(f"  test/unmatched (rule could NOT label)    : {len(unm)}")

    print("\nclass balance in train:")
    for k, v in collections.Counter(r["y"] for r in train).most_common():
        print(f"    {k:12s} {v:6d}")
    print("\nclass balance in unmatched (by model, unknown to the rule):")
    print(f"    all {UNMATCHED}: {len(unm)}")

    for name, rs in (("train", train), ("rule_held", held), ("unmatched", unm)):
        p = os.path.join(ap_out, f"{name}.jsonl")
        with open(p, "w") as f:
            for r in rs:
                f.write(json.dumps({"x": r["x"], "y": r["y"],
                                    "tool": r["tool"], "group": r["group"]}) + "\n")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
