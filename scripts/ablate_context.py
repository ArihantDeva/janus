"""Does the adapter actually READ the context it is given?

The adapter ranks ASB at AUC ~0.55 in every configuration ever tried, while a
TF-IDF bag-of-words over the SAME rendered prompts reaches 0.851 on ASB and 0.974
on R-Judge. So the task is linearly learnable from this text and the adapter is
far below that. Two very different causes fit:

  (a) the adapter reads the context but cannot use it, or
  (b) the adapter ignores the context and answers from the instruction alone.

These are separated by ablation: score each row three ways and compare.
  full   : context + instruction, as trained
  instr  : instruction only, context removed
  rndctx : instruction + a randomly mismatched context (same length distribution)

If `instr` scores ~the same as `full`, the context is being ignored and no amount
of calibration or thresholding will help -- the information never reaches the
decision. If `full` is clearly better than both, the context is being used and
the shortfall is in extracting from it.

This is a diagnostic. It changes nothing that ships.
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
sys.path.insert(0, "/scratch/ad2416/janussrc")

import metrics  # noqa: E402
import evals    # noqa: E402

BASE = "unsloth/Llama-3.2-1B"


def variants(rows):
    """Build the three prompt variants from the already-rendered rows.

    Rows carry the rendered prompt only, so the split is recovered from the
    renderer's own layout: noul.safety_prompt emits Context / Constraints /
    Action sections and the trailing answer cue.
    """
    full, instr, rnd = [], [], []
    ctxs = []
    for r in rows:
        p = r["prompt"]
        # split at the Action: section, which is where the instruction starts
        cut = p.rfind("Action:")
        ctx, act = (p[:cut], p[cut:]) if cut > 0 else ("", p)
        ctxs.append(ctx)
        full.append(p)
        instr.append(act)
    rng = random.Random(0)
    shuffled = ctxs[:]
    rng.shuffle(shuffled)
    for act, c in zip(instr, shuffled):
        rnd.append(c + act)
    return full, instr, rnd


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from train_lora import cand_ids, score
    from train_util import ensure_pad, keep_tail

    adapters = sys.argv[1].split(",")
    maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    ensure_pad(tok); keep_tail(tok)
    cids = cand_ids(tok)

    for bench in ("rjudge", "asb"):
        rows = evals.load_bench(bench)
        y = np.array([r["label"] for r in rows], float)
        fl = float(max(y.mean(), 1 - y.mean()))
        full, instr, rnd = variants(rows)
        print(f"\n=== {bench.upper()}  n={len(y)}  floor {fl*100:.1f}% ===")
        for i, ad in enumerate(adapters):
            base = AutoModelForCausalLM.from_pretrained(
                BASE, torch_dtype=torch.bfloat16).to(dev)
            model = PeftModel.from_pretrained(base, ad)
            out = {}
            for tag, prompts in (("full", full), ("instr", instr), ("rndctx", rnd)):
                p = np.asarray(score(model, tok, prompts, cids, maxlen=maxlen), float)
                out[tag] = (metrics.auc(p, y), float(((p >= 0.5).astype(int) == y).mean()))
            del model, base
            torch.cuda.empty_cache()
            print(f"  seed{i}: " + "  ".join(
                f"{t} auc {out[t][0]:.3f} acc {out[t][1]*100:5.1f}%"
                for t in ("full", "instr", "rndctx")), flush=True)
            print(f"         context gain (full - instr) = "
                  f"{out['full'][0]-out['instr'][0]:+.3f} auc, "
                  f"{(out['full'][1]-out['instr'][1])*100:+.1f}pt acc", flush=True)
    print("\nABLATE_DONE")


if __name__ == "__main__":
    main()
