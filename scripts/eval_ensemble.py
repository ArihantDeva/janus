"""Average predictions across seeds and score the ensemble.

The noise floor measured in this project is sd ~0.02 AUC from a pure seed
change, which is larger than nearly every hyperparameter difference observed.
The textbook response to a noise-limited predictor is not a better single model
but an average of several -- and averaging is FIXED IN ADVANCE here (mean of the
per-seed probabilities, no weights, no selection) so it cannot be tuned to the
benchmark.

No new training: it reuses the adapters already on disk.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
sys.path.insert(0, "/scratch/ad2416/janussrc")

import metrics  # noqa: E402
import evals    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--bench", default="both", choices=("rjudge", "asb", "both"))
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from train_lora import cand_ids, score
    from train_util import ensure_pad, keep_tail

    BASE = "unsloth/Llama-3.2-1B"
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    ensure_pad(tok); keep_tail(tok)
    cids = cand_ids(tok)

    results = []
    for bench in (("rjudge", "asb") if a.bench == "both" else (a.bench,)):
        rows = evals.load_bench(bench)
        prompts = [r["prompt"] for r in rows]
        y = np.array([r["label"] for r in rows], float)

        member_probs = []
        for ad in a.adapters:
            base = AutoModelForCausalLM.from_pretrained(
                BASE, torch_dtype=torch.bfloat16).to(dev)
            model = PeftModel.from_pretrained(base, ad)
            p = score(model, tok, prompts, cids, maxlen=a.maxlen)
            member_probs.append(np.asarray(p, float))
            print(f"  {bench} member {os.path.basename(ad)}: "
                  f"auc {metrics.auc(p, y):.3f} acc {metrics._correct(p, y).mean():.3f}",
                  flush=True)
            del model, base
            torch.cuda.empty_cache()

        # FIXED RULE, chosen before seeing the result: plain arithmetic mean of
        # the per-seed probabilities. No weights, no per-benchmark choice.
        avg = np.mean(np.stack(member_probs), axis=0)
        s = metrics.summarize(avg, y)
        fl = metrics.majority_floor(y) if hasattr(metrics, "majority_floor") else \
            max(y.mean(), 1 - y.mean())
        print(f"\n=== {bench.upper()} ENSEMBLE n={len(y)} "
              f"members={len(a.adapters)} ===")
        print(f"  floor (majority class)   {fl*100:5.1f}%")
        for tag, p in (("ensemble", avg), *[(f"seed{i}", m) for i, m in enumerate(member_probs)]):
            ss = metrics.summarize(p, y)
            print(f"  {tag:10s} acc {ss['acc']*100:5.1f}%  ece {ss['ece']:.4f}  "
                  f"aurc {ss['aurc']:.4f}  lift {(ss['acc']-fl)*100:+5.1f}pt  "
                  f"auc {ss['auc']:.3f}")
        results.append({"bench": bench, "floor": fl,
                        "ensemble": {**s, "auc": metrics.auc(avg, y)},
                        "members": [{**metrics.summarize(m, y),
                                     "auc": metrics.auc(m, y)} for m in member_probs]})

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {a.out}")
    print("ENS_DONE")


if __name__ == "__main__":
    main()
