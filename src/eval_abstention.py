"""Two-stage abstention report for the deployed encoder (I-CALM protocol).

Stage 1: at the operating threshold, how often does the classifier answer,
and how accurate is it on the rows it answers?
Stage 2: on the rows it abstains on, what would its forced best guess have
scored? Abstention is only worth anything if the forced guesses on abstained
rows are much worse than the answered rows — otherwise the threshold is
throwing away correct answers, not routing hard ones.

Dists are computed once (batched, local weights) and saved next to the
report so the numbers can be re-derived without re-running the encoder.

Run: uv run python src/eval_abstention.py [--threshold 0.60] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ("debug", "fix_input", "fix_call")
TEMPERATURE = 0.5276694558560848  # exact fit, eval/temperature_freeze0.json; rescales, never re-ranks


def stage_report(rows, threshold):
    """Pure two-stage summary over (label, dist) rows. No I/O, unit-tested."""
    answered = [(y, d) for y, d in rows if max(d.values()) >= threshold]
    abstained = [(y, d) for y, d in rows if max(d.values()) < threshold]

    def acc(pairs):
        if not pairs:
            return None
        return sum(max(d, key=d.get) == y for y, d in pairs) / len(pairs)

    return {
        "n": len(rows),
        "threshold": threshold,
        "temperature": TEMPERATURE,
        "answered": len(answered),
        "abstained": len(abstained),
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "answered_acc": acc(answered),
        "forced_guess_acc_on_abstained": acc(abstained),
        "overall_forced_acc": acc(rows),
    }


def sweep(rows, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Coverage/accuracy trade-off across thresholds, from the same dists.

    A curve, not an operating point: quoting one threshold's accuracy is how
    a knob gets fitted to a test set (eval_router.py makes the same point
    for the generative tier)."""
    return [stage_report(rows, t) for t in thresholds]


def load_held(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r.get("y") in CLASSES]


def score(rows, weights, head, batch=16, limit=None):
    import torch

    import encoder_backend as EB
    import muse_hook as H

    model, tok = EB.restore(weights, head)
    rows = rows[:limit] if limit else rows
    texts = [H.encoder_input({"tool_name": r.get("tool", ""),
                               "tool_input": r.get("arguments", {}),
                               "error": str(r.get("error", ""))}) for r in rows]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=512)
            logits = model(enc["input_ids"], enc["attention_mask"]).float()
            probs = (logits / TEMPERATURE).softmax(-1).tolist()
            out.extend({c: p for c, p in zip(CLASSES, ps)} for ps in probs)
    return out


def report_from_artifact(path, threshold):
    """Re-derive report + sweep from a saved artifact. No encoder run."""
    saved = json.load(open(path))
    pairs = [(r["y"], r["dist"]) for r in saved["rows"]]
    report = stage_report(pairs, threshold)
    report["sweep"] = sweep(pairs)
    for key in ("held", "weights"):
        if key in saved.get("report", {}):
            report[key] = saved["report"][key]
    saved["report"] = report
    with open(path, "w") as fh:
        json.dump(saved, fh, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-artifact", metavar="PATH",
                    help="re-derive the report from saved dists, no scoring")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--held", default=os.path.join(ROOT, "data/responses_v3/rule_held.jsonl"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights/model_freeze0_rlcd"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/abstention_freeze0.json"))
    a = ap.parse_args()

    if a.from_artifact:
        print(json.dumps(report_from_artifact(a.from_artifact, a.threshold), indent=1))
        return 0

    rows = load_held(a.held)
    dists = score(rows, a.weights, os.path.join(a.weights, "head.pt"), limit=a.limit)
    pairs = [(r["y"], d) for r, d in zip(rows, dists)]
    report = stage_report(pairs, a.threshold)
    report["sweep"] = sweep(pairs)
    report["held"] = os.path.relpath(a.held, ROOT)
    report["weights"] = os.path.relpath(a.weights, ROOT)
    with open(a.out, "w") as fh:
        json.dump({"report": report,
                   "rows": [{"y": y, "dist": d} for y, d in pairs]}, fh, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
