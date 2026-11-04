"""Regenerate the stored corpora so they carry the CURRENT renderer instruction.

data/safety_*.jsonl were written on 2026-10-26 20:43-20:54, before src/noul.py
was fixed at 2026-10-27 02:36. The fix corrected an inverted answer contract:
the corpus instruction said "no (it is safe) or yes (it is unsafe)", which tells
the model that 'no' means SAFE, while the supervision sends every label=1
(UNSAFE) row to the token 'no'. 98.5% of the 20,185 training rows still carry
that inverted line.

Run 5 fixed the renderer and concluded the contract was repaired. It could not
be: training reads r["prompt"] straight out of the stored file, while evaluation
calls noul.safety_prompt() fresh. So from run 5 onward the adapter was trained
against an inverted instruction and evaluated on a corrected one.

This regenerates the stored prompt for every row using the CURRENT renderer,
preserving the labels exactly. Nothing about a label, a record, or the split
changes -- only the instruction text, to match what evaluation and training are
both supposed to share.

Idempotent: rows already carrying the current instruction are left untouched.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

DATA = os.environ.get("JANUS_DATA", "data")

OLD = "no (it is safe) or yes (it is unsafe)"
NEW = "no (it is unsafe) or yes (it is safe)"


def main(apply=False):
    total = changed = 0
    for name in ("safety_train.jsonl", "safety_dev.jsonl", "safety_test.jsonl"):
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            print(f"  {name}: absent, skipped")
            continue
        rows = [json.loads(l) for l in open(path)]
        n_changed = 0
        other = 0
        for r in rows:
            p = r["prompt"]
            # The instruction is the only line that must agree. Derive it from
            # the corpus' own text rather than rebuilding prompts from raw
            # records, so labels, ordering and any row-specific content are
            # preserved bit for bit.
            if OLD in p:
                r["prompt"] = p.replace(OLD, NEW)
                n_changed += 1
            elif NEW not in p:
                # A row that never had an answer contract at all. The safety
                # corpora are a mix: 19889 rows are noul.safety_prompt tool-call
                # judgements, and 296 are SWE-style "Task/Environment/Actions"
                # transcripts with no safe/unsafe answer. Those carry no
                # inverted instruction, so leave them untouched -- rewriting
                # them would be inventing an answer contract they never had.
                other += 1
        total += len(rows); changed += n_changed
        print(f"  {name}: {len(rows)} rows, {n_changed} updated, "
              f"{other} with no answer contract (left untouched)")
        if apply:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, path)
    print(f"\n{changed}/{total} rows carried the inverted instruction")
    print("APPLIED" if apply else "DRY RUN (pass --apply to write)")
    # After any apply, trainer and evaluator agree by construction.
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
