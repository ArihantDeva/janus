"""Build the tool-failure dataset with a leak-free split.

The naive split (shuffle rows, cut 80/10/10) leaks: the same command appears
in many trajectories, so an identical prompt lands in both train and test and
the test score is inflated. The fix is to deduplicate on the prompt and then
assign whole deduplicated prompts to one split each.

Grouping is by prompt text, which is the thing the model actually sees.
"""
import argparse, json, os, random, sys
from collections import Counter

D = os.environ.get("JANUS_DATA",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"))

import noul

INSTRUCTION = noul.INSTRUCTION


def make_prompt(command):
    return f"{INSTRUCTION}\n<command>\n{command}\n</command>\n\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=f"{D}/swe_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--dev", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pairs)]
    # dedupe by prompt; keep the first label seen, count disagreements so the
    # label noise is visible rather than hidden
    by_prompt, conflict = {}, 0
    for r in rows:
        p = make_prompt(r["command"])
        lab = 1 if r["is_error"] else 0
        if p in by_prompt:
            if by_prompt[p]["label"] != lab:
                conflict += 1
            continue
        by_prompt[p] = {"prompt": p, "label": lab,
                        "cls": r["failure_class"], "source": r["source"]}
    uniq = list(by_prompt.values())
    random.Random(a.seed).shuffle(uniq)

    n = len(uniq)
    n_test = int(n * a.test)
    n_dev = int(n * a.dev)
    splits = {"test": uniq[:n_test],
              "dev": uniq[n_test:n_test + n_dev],
              "train": uniq[n_test + n_dev:]}

    for name, part in splits.items():
        path = f"{D}/swe_{name}.jsonl"
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        fail = sum(r["label"] for r in part)
        print(f"{name:5s} {len(part):6d}  fail={fail:5d} ({fail/len(part)*100:5.1f}%)")

    print(f"\nraw pairs        {len(rows)}")
    print(f"unique prompts   {n}  ({len(rows) - n} duplicates collapsed)")
    print(f"label conflicts  {conflict}  "
          f"(same command, both outcomes: irreducible noise)")
    print(f"classes          {dict(Counter(r['cls'] for r in uniq).most_common())}")


if __name__ == "__main__":
    main()
