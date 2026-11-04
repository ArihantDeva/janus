"""Upper bound on what ANY decision threshold could achieve.

Run 2 concluded "no calibration of this adapter clears the floor on these
benchmarks" using an oracle (T,b) fit that reached acc 0.594 vs a 0.596 floor.
That conclusion is stale: it was measured at AUC 0.577 through an evaluator
that truncated 12-15% of rows, before five correctness fixes.

This recomputes the ceiling. For each seed and benchmark it asks: sweeping EVERY
threshold on the predicted probability, what is the best accuracy reachable?
That number is an UPPER BOUND for any thresholding, calibration layer or
decision rule applied to this adapter's scores.

It is a diagnostic, never a shipped configuration: the threshold is chosen on
the benchmark itself, which is exactly why the result can only ever be used to
rule things OUT. If even the oracle cannot beat the majority-class floor, no
amount of threshold tuning, temperature fitting or operating-point selection
will, and the loop should stop rather than search that space.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
sys.path.insert(0, "/scratch/ad2416/janussrc")

import metrics  # noqa: E402
import evals    # noqa: E402

BASE = "unsloth/Llama-3.2-1B"


def oracle_threshold(p, y):
    """Best accuracy over all thresholds, and the threshold achieving it."""
    p = np.asarray(p, float)
    cand = np.unique(np.concatenate([p, np.nextafter(p, np.inf)]))
    best = (-1.0, 0.5)
    for t in cand:
        acc = float((((p >= t).astype(int)) == y.astype(int)).mean())
        if acc > best[0]:
            best = (acc, float(t))
    return best


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from train_lora import cand_ids, score
    from train_util import ensure_pad, keep_tail

    adapters = sys.argv[1].split(",")
    out_path = sys.argv[2]
    maxlen = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    ensure_pad(tok); keep_tail(tok)
    cids = cand_ids(tok)

    store = {}
    for bench in ("rjudge", "asb"):
        rows = evals.load_bench(bench)
        prompts = [r["prompt"] for r in rows]
        y = np.array([r["label"] for r in rows], float)
        for i, ad in enumerate(adapters):
            base = AutoModelForCausalLM.from_pretrained(
                BASE, torch_dtype=torch.bfloat16).to(dev)
            model = PeftModel.from_pretrained(base, ad)
            p = np.asarray(score(model, tok, prompts, cids, maxlen=maxlen), float)
            store[f"{bench}_s{i}_p"] = p
            store[f"{bench}_s{i}_y"] = y
            del model, base
            torch.cuda.empty_cache()
            print(f"scored {bench} seed{i}", flush=True)

    np.savez(out_path, **store)
    fl = float(max(y.mean(), 1 - y.mean()))
    print(f"\n=== ORACLE THRESHOLD CEILING (saved to {out_path}) ===")
    for bench in ("rjudge", "asb"):
        print(f"\n{bench.upper()}")
        floors, accs = [], []
        for i in range(len(adapters)):
            p = store[f"{bench}_s{i}_p"]; y = store[f"{bench}_s{i}_y"]
            f = float(max(y.mean(), 1 - y.mean()))
            oa, ot = oracle_threshold(p, y)
            a05 = float((((p >= 0.05).astype(int)) == y.astype(int)).mean())
            floors.append(f); accs.append(oa)
            print(f"  seed{i}: floor {f*100:5.1f}%  oracle acc {oa*100:5.1f}% "
                  f"(t={ot:.3f})  lift {(oa-f)*100:+5.1f}pt  |  acc@t=0.05 {a05*100:5.1f}%")
        fl = float(np.mean(floors))
        m, sd = float(np.mean(accs)), float(np.std(accs, ddof=1))
        se = sd / len(accs) ** 0.5
        t = (m - fl) / se if se > 0 else float("inf")
        print(f"  ORACLE mean {m*100:5.1f}% +- {sd*100:4.1f}  vs floor {fl*100:5.1f}%"
              f"  -> lift {(m-fl)*100:+5.1f}pt  t={t:+.2f}")
        verdict = ("REACHABLE by some threshold" if m > fl and abs(t) > 2
                   else "NOT reachable -- no threshold clears the floor")
        print(f"  => {verdict}")
    print("\nORACLE_DONE")


if __name__ == "__main__":
    main()
