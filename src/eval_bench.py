"""Score a saved adapter on R-Judge and Agent-SafetyBench.

Reports the three configurations the goal requires (zero-shot, fine-tuned,
fine-tuned + temperature) on each benchmark, with the trivial-baseline floor
printed beside every number.

Two honesty rules are enforced in the output rather than in prose:

  * the floor is always printed, because R-Judge is separable by surface
    statistics (+18.5 pts at action level) and a bare accuracy there would be
    misleading
  * the temperature is fitted on a held-out split and reported on the other, so
    the calibrated number is not an in-sample optimum

    python -u src/eval_bench.py --adapter out/adapter_seed0 --bench asb
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evals  # noqa: E402
from train_util import ensure_pad, keep_tail
from metrics import (apply_temperature, fit_temperature,  # noqa: E402
                     fit_temperature_info, gate, summarize, logit as _logit)
from train_lora import cand_ids, score  # noqa: E402

# Overridable so a different base model can be evaluated through the SAME
# pipeline. Default is unchanged, so existing results stay reproducible.
BASE = os.environ.get("JANUS_BASE_MODEL", "unsloth/Llama-3.2-1B")
#: Must match the trainer's --maxlen. It did not: this defaulted to 192 while
#: run_loop_iter.sh trained at 512 and never passed --maxlen through, so every
#: evaluation scored 12-16% of rows truncated while the adapter had been trained
#: on the full prompt. Fixing only the runner leaves the trap for the next caller,
#: so the default is aligned too.
MAXLEN = 512
DATA = os.environ.get("JANUS_DATA", "data")


def floor(rows):
    y = np.array([r["label"] for r in rows])
    return float(max(y.mean(), 1 - y.mean()))


def load_dev_rows():
    """Rows used to FIT the temperature: the model's own dev split.

    Fitting T on the benchmark itself would leak the benchmark into the
    calibration curve. Fitting it on half the benchmark and reporting the other
    half is honest but throws away half the evaluation. The dev split exists
    precisely to be spent on this, so it is spent. Mean of 3 seeds is used when
    several are present, because a single seed's optimum is noise.
    """
    # The dev split MUST be the corpus the model was trained on. This used to
    # hardcode the SWE dev split while the safety run trained on the safety
    # corpora, so temperature and the intercept were fitted on a DIFFERENT TASK:
    # 30.5%-unsafe "coding agent" prompts calibrating a model evaluated on
    # 54.0%-unsafe "autonomous agent" prompts. That is how the fit picked a
    # strongly negative intercept and produced predict-all-safe, while the oracle
    # on the real rows wanted a positive one.
    corpus = os.environ.get("JANUS_CORPUS", "swe")
    path = f"{DATA}/{corpus}_dev.jsonl"
    if not os.path.exists(path):
        raise SystemExit(
            f"dev split {path} is missing; cannot calibrate. Point JANUS_CORPUS at "
            f"the corpus the adapter was trained on (currently {corpus!r}).")
    with open(path) as f:
        return [json.loads(l) for l in f]


def run(adapter, bench, maxlen=MAXLEN, report_full=True,
        prior_shift=0.0, plain=False):
    rows = evals.load_bench(bench)
    tok = AutoTokenizer.from_pretrained(BASE)
    ensure_pad(tok)
    keep_tail(tok)
    cids = cand_ids(tok)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16).to(dev)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base

    if report_full:
        # Report the WHOLE benchmark with T fitted on the model's own dev
        # split, which is what that split is for.
        rep_rows = rows
        fit_rows = load_dev_rows()
    else:
        a, b = evals.split_rows(rows, 0.5, seed=0)
        fit_rows, rep_rows = (a, b) if len(a) >= len(b) else (b, a)

    yv = np.array([r["label"] for r in fit_rows])
    yr = np.array([r["label"] for r in rep_rows])
    pv = score(model, tok, [r["prompt"] for r in fit_rows], cids, maxlen=maxlen)
    tp = score(model, tok, [r["prompt"] for r in rep_rows], cids, maxlen=maxlen)

    # The zero-shot baseline must come from the BASE weights. With an adapter
    # attached that means explicitly switching it off; scoring it while the
    # adapter is live would silently report the fine-tuned model as "baseline".
    if adapter:
        with model.disable_adapter():
            zp = score(model, tok, [r["prompt"] for r in rep_rows], cids,
                       maxlen=maxlen)
    else:
        zp = tp

    # Report truncation rather than swallowing it: a T sitting on a bound means
    # the fit did not converge, so the calibration that follows is a property of
    # the search grid, not of the model.
    if adapter:
        info = fit_temperature_info(pv, yv)
        T = info["T"]
    else:
        info, T = {"at_bound": False, "nll": float("nan")}, 1.0
    if info["at_bound"]:
        print(f"  WARNING: T={T:.2f} sits on a search bound; the fit did not "
              f"converge and this calibration is truncated.", flush=True)
        # Name the direction. A ceiling hit means the probabilities carry less
        # information than the LABELS do, which is a statement about the model,
        # not about the search range -- raising the ceiling further (as was
        # already done once, 20 -> 200) cannot fix it.
        if info.get("at_ceiling"):
            print(f"  CEILING_HIT: the optimum is 'as flat as possible'. The "
                  f"model's ranking may still be informative (see corr) even "
                  f"though its probabilities are not. Reported ECE below is a "
                  f"consequence of near-uniform output, not a calibration win.",
                  flush=True)

    # Joint (T, b) fit, with a TRANSFER GUARD.
    #
    # b is fitted on the model's dev split, but dev and the benchmark are
    # different populations: fitting b on dev produced -0.50 and then HURT the
    # benchmark badly (rjudge 65.5% -> 62.3%, asb 59.5% -> 49.8%). A level
    # correction that does not transfer is worse than no correction, so b is
    # only kept if it also improves log-loss on a HELD-OUT half of the reported
    # rows; otherwise b = 0 and the shipped configuration is plain temperature.
    #
    # The held-out half is used only to accept or reject the correction, never
    # to fit it, so this is a gate rather than a second fitting pass.
    b, Tj = 0.0, T
    if adapter:
        lg_raw = _logit(pv)
        best = (np.inf, T, 0.0)
        for tc in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0):
            lgt = lg_raw / tc
            for bc in np.arange(-2.0, 2.0 + 1e-9, 0.1):
                q = np.clip(1.0 / (1.0 + np.exp(-(lgt + bc))), 1e-7, 1 - 1e-7)
                nll = -np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q))
                if nll < best[0]:
                    best = (float(nll), float(tc), float(bc))
        _nll, Tj, b = best

        # transfer guard on a deterministic hold-out half of the report rows
        n_rep = len(yr)
        hold = np.arange(n_rep) % 2 == 1
        if hold.sum() and (~hold).sum():
            lg_rep = _logit(tp) / Tj
            def _nll_rep(bb):
                q = np.clip(1.0 / (1.0 + np.exp(-(lg_rep + bb))), 1e-7, 1 - 1e-7)
                return float(-np.mean(yr[hold] * np.log(q[hold])
                                      + (1 - yr[hold]) * np.log(1 - q[hold])))
            if b != 0.0 and _nll_rep(b) >= _nll_rep(0.0):
                print(f"  TRANSFER_GUARD: dev-fitted b={b:+.2f} does not improve "
                      f"held-out log-loss ({_nll_rep(b):.4f} vs "
                      f"{_nll_rep(0.0):.4f}); dropping it", flush=True)
                b = 0.0

    # Keep the temperature-only row for comparison: it is the grandfathered
    # configuration every earlier run was measured under.
    tpc_only = apply_temperature(tp, T)
    tpc = apply_temperature(tp, Tj)
    lg = _logit(tpc)
    tpb = 1.0 / (1.0 + np.exp(-(lg + b)))

    fl = floor(rows)
    print(f"\n=== {bench.upper()}  n_report={len(rep_rows)} "
          f"fail={int(yr.sum())}  T={Tj:.2f} b={b:+.2f} "
          f"(T fitted on {len(fit_rows)} dev rows) ===")
    print(f"  floor (majority class)   {fl*100:5.1f}%   "
          f"<- a constant predictor scores this")
    out = {"bench": bench, "adapter": adapter, "T": Tj, "b": b,
           "T_temperature_only": T,
           "T_at_bound": info["at_bound"],
           "T_at_ceiling": bool(info.get("at_ceiling", False)),
           "n_report": len(rep_rows), "n_fit": len(fit_rows), "floor": fl}
    rows_to_report = [("zero-shot", zp), ("finetuned", tp),
                      ("finetuned+T", tpc_only)]
    if not plain:
        rows_to_report.append(("finetuned+T+b", tpb))
    for tag, p in rows_to_report:
        s = summarize(p, yr)
        s["corr"] = float(np.corrcoef(p, yr)[0, 1]) if p.std() > 0 else 0.0
        s["gate_0.7_0.3"] = gate(p, yr, 0.7, 0.3)
        out[tag] = s
        print(f"  {tag:12s} acc {s['acc']*100:5.1f}%  brier {s['brier']:.4f}  "
              f"ece {s['ece']:.4f}  aurc {s['aurc']:.4f}  "
              f"cov@5 {s['cov5']*100:3.0f}%  "
              f"lift {(s['acc']-fl)*100:+5.1f}pt  corr {s['corr']:+.3f}  "
              f"auc {s['auc']:.3f}")

    # Per-row predictions, so calibration and threshold questions can be asked
    # later WITHOUT re-running the model. Every expensive number so far raised a
    # follow-up question ("is T at the bound real?", "what accuracy would a prior
    # shift give?", "does a floor-rounded threshold help?") and each one cost a
    # GPU-hour to answer. This makes all of them free from here on.
    out["rows"] = {"p_shifted": [float(v) for v in tpb],
                   "p_zero_shot": [float(v) for v in zp],
                   "p_finetuned": [float(v) for v in tp],
                   "p_finetuned_T": [float(v) for v in tpc],
                   "y": [int(v) for v in yr]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bench", default="both",
                    choices=("rjudge", "asb", "both"))
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prior-shift", type=float, default=0.0,
                    help="intercept correction in logit space, fitted on dev. "
                         "Temperature can only pull toward 0.5 and so cannot "
                         "repair a LEVEL error; this can.")
    ap.add_argument("--plain", action="store_true",
                    help="report only the shipped (T-only) configuration")
    ap.add_argument("--half", action="store_true",
                    help="fit T on half the benchmark instead of "
                         "the model dev split (reports half)")
    a = ap.parse_args()

    benches = ("rjudge", "asb") if a.bench == "both" else (a.bench,)
    res = [run(a.adapter, b, a.maxlen, report_full=not a.half,
               prior_shift=a.prior_shift, plain=a.plain)
           for b in benches]
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
