"""Turn real Janus usage into a calibration set.

WHAT THIS IS
------------
The user's own sessions contain 42 real `janus_ask` calls: the state and typed
questions that were actually sent, and the calibrated probabilities that came
back. That is live supervision from the model this project is trying to
reproduce -- not synthetic data, not benchmark rows.

WHY IT IS NOT A TRAINING SET
----------------------------
42 calls contain 72 answerable decisions across 26 distinct states. Training a
1B LoRA on 72 decisions would memorise them. Their value is as a MEASUREMENT:
held-out comparisons where a candidate model is scored against the reference
probabilities Janus actually produced, on inputs nobody picked to be easy.

So this is used in the role the benchmarks were filling, but honestly: a small,
real, in-domain held-out set. It answers "does our model's distribution resemble
Janus's on the kind of input Janus is actually used on?" -- a different question from
"is it right", and one the benchmarks cannot answer at all because they are a
different task shape.

THE HONEST LIMITS, STATED UP FRONT
----------------------------------
- n=42 calls / 72 decisions. Enough for a direction, not for a claim of a few
  points. Any number from it is reported with its n.
- The reference is Janus 1.13.0/1.14.0, not ground truth. Agreement with Janus is
  agreement with a model, and the open-janus review already noted that distilling
  a teacher inherits the teacher's biases.
- The questions vary by session, so decisions are not i.i.d. across a shared
  task. Grouping by call is required when computing any interval.
- States contain the user's real work. They are used here for local measurement
  only and are never sent to a remote node or written into a corpus used for
  training a model that ships.

WHAT IS EXTRACTED
-----------------
Per decision: the question text, the answer type, the FULL reference
distribution (not just the argmax -- the distribution is the calibration signal),
and the answer Janus picked.
"""

from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_result(text):
    """Parse Janus's textual result into per-question reference distributions.

    Two answer shapes appear in real logs and both are handled:

        noul   `name: yes 0.92  <- question`
        choice `name: chosen (conf 0.38) [a 0.53, b 0.44, c 0.02, d 0.01]  <- question`
        score  `name: 1.78/3 (conf 0.73)  <- question`

    Parsed by pattern rather than by splitting on ':' because question text
    itself contains colons, which a naive split would cut in the wrong place.
    Returns a list of dicts; a line that matches nothing is DROPPED and counted,
    never silently guessed at.
    """
    out, dropped = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("model janus") or line.startswith("tokens"):
            continue

        m = re.match(r"^(.+?):\s*(yes|no)\s+([01](?:\.\d+)?)\s*(?:<- (.*))?$", line)
        if m:
            name, word, p, q = m.groups()
            pv = float(p)
            # "yes 0.92" means P(yes)=0.92; the distribution is the pair.
            out.append({"id": name, "type": "noul", "question": (q or "").strip(),
                        "probs": [1.0 - pv, pv], "labels": ["no", "yes"]})
            continue

        m = re.match(r"^(.+?):\s*(.*?)\(conf\s*([\d.]+)\)\s*\[(.+?)\]\s*(?:<- (.*))?$", line)
        if m:
            name, chosen, conf, dist, q = m.groups()
            # Option labels routinely contain SPACES ("Deep Tech",
            # "Prototype stage, pre-launch"). A \w-only pattern silently keeps
            # just the last word, which produced a label that disagreed with the
            # stated choice on 3/44 rows. Greedy non-greedy split on the final
            # number is what actually separates label from probability.
            # Option labels contain SPACES and COMMAS ("Prototype stage,
            # pre-launch, no users, no revenue"). Splitting on either separator
            # truncates them. A probability is a number at the very end of a
            # segment, so split on the number itself and let everything before it
            # be the label.
            pairs = re.findall(r"([^\[\]]*?)\s+([\d.]+)(?=\s*(?:,|$))", dist)
            if not pairs:
                dropped += 1
                continue
            out.append({"id": name, "type": "choice", "question": (q or "").strip(),
                        "chosen": chosen.strip(),
                        "probs": [float(v) for _, v in pairs],
                        "labels": [k.strip(" ,") for k, _ in pairs]})
            continue

        m = re.match(r"^(.+?):\s*([\d.]+)\s*/\s*(\d+)\s*\(conf\s*([\d.]+)\)\s*(?:<- (.*))?$", line)
        if m:
            name, val, hi, conf, q = m.groups()
            out.append({"id": name, "type": "score", "question": (q or "").strip(),
                        "value": float(val), "max": int(hi), "conf": float(conf)})
            continue

        dropped += 1
    return out, dropped


def extract(pairs_path="/tmp/janus_pairs.json"):
    pairs = json.load(open(pairs_path))
    records, dropped_total = [], 0
    for p in pairs:
        parsed, dropped = parse_result(p["result"])
        dropped_total += dropped
        args = p.get("args") or {}
        for q in parsed:
            q["state"] = args.get("state", "")
            q["ts"] = p.get("ts")
            records.append(q)
    return records, dropped_total


def summarise(records):
    import collections
    by_type = collections.Counter(r["type"] for r in records)
    by_day = collections.Counter(str(r["ts"])[:10] for r in records)
    check = [r for r in records if r["type"] in ("choice", "noul")]
    bad_sum = [r for r in check if abs(sum(r["probs"]) - 1.0) > 0.02]
    return {"n_decisions": len(records), "by_type": dict(by_type),
            "days": len(by_day), "non_normalised": len(bad_sum)}


def main():
    recs, dropped = extract()
    s = summarise(recs)
    print(f"decisions extracted : {s['n_decisions']}")
    print(f"by type             : {s['by_type']}")
    print(f"distinct days       : {s['days']}")
    print(f"lines not parsed    : {dropped}")
    print(f"distributions not summing to 1 (+-0.02): {s['non_normalised']}")
    out = os.path.join(ROOT, "data", "janus_live", "decisions.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(recs, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print("\nNOTE: states are the user's real work. Local measurement only -- "
          "never sent to a remote node, never trained into a shipped model.")


if __name__ == "__main__":
    main()
