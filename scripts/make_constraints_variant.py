"""Build a corpus variant with the Constraints section blanked.

Diagnostic basis (run 22): a linear model trained on our corpus keys on
constraint vocabulary -- "never", "without explicit", "permissions never" for
unsafe, "permitted", "ensure", "sandbox" for safe -- because the corpora are
synthetic permission prose. Blanking Constraints in training lifted benchmark
transfer from auc 0.483 to 0.670 at unchanged own-dev (0.893 -> 0.893).

HONEST CAVEAT, recorded because it matters: the reason we know to do this is
that both benchmarks happen to carry an empty Constraints section (0/944 and
0/1994 rows). So this is a benchmark-INFORMED corpus change. It is defensible as
confound removal -- the field is a spurious shortcut, and removing it forces the
model onto the action itself -- but it must not be presented as a benchmark-free
choice, and any gain it produces is a property of a corpus built with knowledge
of the eval distribution.

Rows keep their labels and split exactly. Only the Constraints body is replaced,
with the same placeholder the renderer emits when a constraint is absent
("- (none stated)"), so the prompt SHAPE the model sees is unchanged.
"""
import json
import os
import sys

DATA = os.environ.get("JANUS_DATA", "data")
OUT = os.environ.get("JANUS_OUTDIR", os.path.join(DATA, "noconstraints"))
PLACEHOLDER = "- (none stated)"


def blank_constraints(prompt):
    """Replace the Constraints body with the empty placeholder."""
    lines = prompt.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        out.append(line)
        if line.strip() == "Constraints:":
            i += 1
            # consume the bullet block that follows
            while i < n and lines[i].lstrip().startswith("- "):
                i += 1
            out.append(PLACEHOLDER)
            continue
        i += 1
    return "\n".join(out)


def main(apply=False):
    os.makedirs(OUT, exist_ok=True)
    for name in ("safety_train.jsonl", "safety_dev.jsonl", "safety_test.jsonl"):
        src = os.path.join(DATA, name)
        if not os.path.exists(src):
            print(f"  {name}: absent, skipped")
            continue
        rows = [json.loads(l) for l in open(src)]
        changed = 0
        for r in rows:
            if "exactly one word" not in r["prompt"]:
                continue          # SWE-style row, different task, leave alone
            new = blank_constraints(r["prompt"])
            if new != r["prompt"]:
                r["prompt"] = new
                changed += 1
        print(f"  {name}: {len(rows)} rows, {changed} constraints blanked")
        if apply:
            with open(os.path.join(OUT, name), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    print("APPLIED to " + OUT if apply else "DRY RUN (pass --apply)")


if __name__ == "__main__":
    main("--apply" in sys.argv)
