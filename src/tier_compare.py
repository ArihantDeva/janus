"""Cheaper-and-faster case: same failures, both tiers, timed + priced.

For each sampled held failure: (1) Janus local verdict via the deployed
encoder through the hook's own input path + wall-clock it; (2) Muse Spark
frontier verdict via janus_route.escalate + wall-clock it + record usage.
Reports agreement between tiers, median latency ratio, and token cost ratio.

Writes eval/tier_compare.json. Frontier calls cost real money — keep N small
(default 12) and sampled deterministically (every k-th held row).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ("debug", "fix_input", "fix_call")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--held", default=os.path.join(ROOT, "data/responses_v3/rule_held.jsonl"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights/model_freeze0_rlcd"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/tier_compare.json"))
    a = ap.parse_args()

    import janus_route as R
    import muse_hook as H

    rows = [json.loads(l) for l in open(a.held) if l.strip()]
    rows = [r for r in rows if r.get("y") in CLASSES]
    step = max(1, len(rows) // a.n)
    rows = rows[::step][:a.n]
    print(f"sampled {len(rows)} held failures (every {step}th)", flush=True)

    import torch
    import encoder_backend as EB
    model, tok = EB.restore(a.weights, os.path.join(a.weights, "head.pt"))
    model.eval()

    results = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            text = H.encoder_input({"tool_name": r.get("tool", ""),
                                    "tool_input": r.get("arguments", {}),
                                    "error": str(r.get("error", ""))})
            t0 = time.time()
            enc = tok([text], return_tensors="pt", truncation=True, max_length=512)
            logits = model(enc["input_ids"], enc["attention_mask"]).float()
            probs = (logits / 0.5276694558560848).softmax(-1)[0].tolist()
            local_ms = (time.time() - t0) * 1000
            best = CLASSES[int(logits.argmax())]
            t0 = time.time()
            lab, raw, use = R.escalate(text)
            frontier_ms = (time.time() - t0) * 1000
            results.append({"y": r["y"], "local": best,
                            "local_conf": round(max(probs), 3),
                            "local_ms": round(local_ms, 1),
                            "frontier": lab, "frontier_ms": round(frontier_ms, 1),
                            "reasoning_tokens": use["reasoning_tokens"],
                            "total_tokens": use["total_tokens"]})
            print(f"  {i+1}/{len(rows)} local={best} frontier={lab} "
                  f"{local_ms:.0f}ms vs {frontier_ms:.0f}ms", flush=True)
    agree = sum(r["local"] == r["frontier"] for r in results if r["frontier"]) / max(1, sum(1 for r in results if r["frontier"]))
    lms = sorted(r["local_ms"] for r in results)
    fms = sorted(r["frontier_ms"] for r in results)
    report = {"n": len(results), "tier_agreement": round(agree, 3),
              "local_ms_p50": lms[len(lms)//2],
              "frontier_ms_p50": fms[len(fms)//2],
              "frontier_reasoning_tokens_p50": sorted(r["reasoning_tokens"] for r in results)[len(results)//2],
              "rows": results}
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
