"""Render the user's original goal into the stored corpus.

The karanxa corpus carries `original_goal` on 15134 of 26731 rows (56.6%) and the
renderer never used it, so the model never saw the request the action is supposed
to serve. That is the same omission class as the stale-instruction bug: a field
that exists, is informative, and is silently dropped.

A linear probe suggested this helps transfer (rjudge auc 0.523 -> 0.588 at
unchanged own-val 0.939), but run 23 showed a probe direction does NOT predict
adapter behaviour, so the probe is only a reason to test, not evidence.

Rows without a goal, and rows with no answer contract, are left byte-identical.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

DATA = os.environ.get("JANUS_DATA", "data")
OUT = os.environ.get("JANUS_OUTDIR", os.path.join(DATA, "withgoal"))


def action_of(prompt):
    i = prompt.find("\nAction: ")
    return prompt[i + len("\nAction: "):].split("\n")[0] if i >= 0 else None


def add_goal(prompt, goal):
    """Insert a Goal line before the Action line, keeping the shape otherwise."""
    if not goal or "Goal:" in prompt:
        return prompt
    marker = "\nAction: "
    i = prompt.find(marker)
    if i < 0:
        return prompt
    return prompt[:i] + f"\nGoal: {goal}" + prompt[i:]


def main(apply=False):
    # Corpus rows carry no id, so join on the ACTION text. 26612 of 26731 actions
    # are unique; the 77 duplicates include 64 with conflicting goals, so those
    # are skipped rather than guessed -- assigning the wrong goal would be
    # inventing supervision, which this project forbids.
    src_rows = [json.loads(l) for l in open(os.path.join(DATA, "karanxa.jsonl"))]
    goals = {}
    ambiguous = set()
    for r in src_rows:
        a = r.get("action")
        if not a:
            continue
        g = r.get("original_goal")
        if a in goals and goals[a] != g:
            ambiguous.add(a)
        else:
            goals[a] = g
    for a in ambiguous:
        goals.pop(a, None)

    os.makedirs(OUT, exist_ok=True)
    for name in ("safety_train.jsonl", "safety_dev.jsonl", "safety_test.jsonl"):
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        changed = 0
        for r in rows:
            if r.get("src") != "karanxa":
                continue
            a = action_of(r["prompt"])
            g = goals.get(a)
            if not g:
                continue
            new = add_goal(r["prompt"], g)
            if new != r["prompt"]:
                r["prompt"] = new
                changed += 1
        print(f"  {name}: {len(rows)} rows, {changed} given a Goal line")
        if apply:
            with open(os.path.join(OUT, name), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    print(("APPLIED to " + OUT) if apply else "DRY RUN (pass --apply)")


if __name__ == "__main__":
    main("--apply" in sys.argv)
