"""Is the R-Judge result safety discrimination, or just prompt length?

Run 20 measured length-only features (chars, newlines, spaces) at auc 0.776 /
acc 71.7% on rjudge -- ABOVE the adapter's 0.645 / 65.8%. The project README
also pre-registered this artefact before any GPU was spent. It has been carried
as a caveat twice and never actually tested against the adapter.

This audits the one positive result of the session. Three measurements:

1. LENGTH-ONLY baseline on the same rows: how much does length alone buy?
2. LENGTH-STRATIFIED auc: split the rows into length deciles and compute auc
   WITHIN each. Within a decile, length is nearly constant, so any discrimination
   there cannot be length. If within-decile auc collapses toward 0.5, the
   adapter's overall auc is a length artefact. If it holds near the overall value,
   the signal is content.
3. PARTIAL CORRELATION of prediction with label, controlling for length rank.

Reported per seed so it is subject to the same n=3 discipline as everything else.
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


def length_feats(prompts):
    return np.array([[len(p), p.count("\n"), p.count(" ")] for p in prompts], float)


def length_only_auc(prompts, y):
    """auc of a rule that sees ONLY length features.

    No sklearn on the cluster, and a fitted length model would need CV anyway.
    The honest simple version: score by character count alone. That is the
    strongest single length feature and needs no fitting, so nothing is tuned.
    """
    L = length_feats(prompts)[:, 0]
    a = metrics.auc(L, y)              # longer => higher score
    a2 = metrics.auc(-L, y)            # or shorter => higher score
    if a2 > a:
        p = -L; a = a2
    else:
        p = L
    # best operating point on length alone, for an accuracy figure
    best = 0.0
    for t in np.unique(p):
        best = max(best, float((((p >= t).astype(int)) == y).mean()))
    return float(a), best


def within_strata_auc(p, y, lengths, n_strata=10):
    """auc computed INSIDE length deciles, so length is ~constant."""
    order = np.argsort(lengths)
    aucs, weights = [], []
    for chunk in np.array_split(order, n_strata):
        yy = y[chunk]
        if len(np.unique(yy)) < 2:
            continue
        a = metrics.auc(p[chunk], yy)
        aucs.append(a); weights.append(len(chunk))
    if not aucs:
        return float("nan"), 0
    return float(np.average(aucs, weights=weights)), len(aucs)


def partial_corr(x, y, z):
    """corr(x, y) with z (length rank) regressed out of both."""
    z = z - z.mean()
    def resid(v):
        beta = (v * z).sum() / (z * z).sum()
        return v - beta * z
    return float(np.corrcoef(resid(x), resid(y))[0, 1])


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

    rows = evals.load_bench("rjudge")
    prompts = [r["prompt"] for r in rows]
    y = np.array([r["label"] for r in rows], float)
    lens = length_feats(prompts)
    fl = float(max(y.mean(), 1 - y.mean()))

    l_auc, l_acc = length_only_auc(prompts, y)
    print(f"\n=== RJUDGE  n={len(y)}  floor {fl*100:.1f}% ===")
    print(f"LENGTH-ONLY baseline: auc {l_auc:.3f}  acc {l_acc*100:.1f}%  "
          f"lift {(l_acc-fl)*100:+.1f}pt", flush=True)

    print(f"\n{'seed':6s} {'adapter auc':>12s} {'within-decile auc':>19s} "
          f"{'partial corr':>13s}  verdict")
    for i, ad in enumerate(adapters):
        base = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16).to(dev)
        model = PeftModel.from_pretrained(base, ad)
        p = np.asarray(score(model, tok, prompts, cids, maxlen=maxlen), float)
        del model, base
        torch.cuda.empty_cache()

        overall = metrics.auc(p, y)
        within, nstr = within_strata_auc(p, y, lens[:, 0])
        pc = partial_corr(p, y, lens[:, 0].astype(float))
        # a length artefact would show overall auc >> within-decile auc
        drop = overall - within
        verdict = ("LENGTH ARTEFACT" if within < 0.55
                   else "content signal survives" if within > overall - 0.03
                   else "partly length")
        print(f"{i:<6d} {overall:12.3f} {within:19.3f} {pc:13.3f}  {verdict}",
              flush=True)
        print(f"       ({nstr} deciles with both classes; "
              f"overall - within = {drop:+.3f})", flush=True)
    print("\nLENGTH_DONE")


if __name__ == "__main__":
    main()
