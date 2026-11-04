"""Rescore saved Glimmer QLoRA on held-out splits, then fit temperature.

GPU rescores (logits); CPU could fit T, but one job keeps provenance clean:
logits -> calib-only T fit -> ECE before/after on EVAL -> JSON artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import janus3 as J3  # noqa: E402
import janus4_data as JD  # noqa: E402
import janus4_args as JA  # noqa: E402
import janus4_score as JS  # noqa: E402
from calibrate import (apply_temperature, expected_calibration_error,
                       assert_disjoint, fit_temperature)  # noqa: E402
from train_glimmer_janus import build_model, cand_ids  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = os.environ.get("JANUS_MODEL", "meta-models/Muse-Glimmer-30B")


def split_three(rows, eval_frac, seed):
    tr, ev = JD.split_groups(rows, eval_frac, seed)
    tra, cal = JD.split_groups(tr, 0.15, seed + 500)
    assert_disjoint(
        [{"group": r[3]} for r in tra],
        [{"group": r[3]} for r in cal],
        [{"group": r[3]} for r in ev])
    return tra, cal, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--corpus", default=os.environ.get(
        "JANUS_CORPUS", JA.CORPUS))
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="",
                      help="slice ROWS as start:end (0-based) over the "
                      "calib+eval pool; saves logits only, no T fit")
    a = ap.parse_args()

    import torch
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cids = cand_ids(tok)

    rows = JD.load_rows(a.corpus)
    _, cal, ev = split_three(rows, a.eval_frac, a.seed)
    print(f"calib {len(cal)} / eval {len(ev)}", flush=True)
    pool = cal + ev
    if a.rows:
        lo, hi = (int(x) for x in a.rows.split(":"))
        part = pool[lo:hi]
        is_cal = [i < len(cal) for i in range(lo, min(hi, len(pool)))]
        print(f"logits-only rows {lo}:{hi} n={len(part)}", flush=True)
        model = build_model(16, 32, resume_from=a.adapter)
        model.eval()
        P = JS.score(model, tok, [p for p, _, _, _ in part],
                     cids, maxlen=a.maxlen)
        y = np.array([y for _, _, y, _ in part])
        os.makedirs(a.out, exist_ok=True)
        np.savez(os.path.join(a.out, f"logits_{lo}_{hi}.npz"),
                 P=P, y=y, is_cal=np.array(is_cal))
        print(f"saved logits_{lo}_{hi}.npz", flush=True)
        return

    model = build_model(16, 32, resume_from=a.adapter)
    model.eval()

    def logits_for(split):
        P = JS.score(model, tok, [p for p, _, _, _ in split],
                     cids, maxlen=a.maxlen)
        return np.log(np.clip(P, 1e-12, None))

    Lc = logits_for(cal)
    yc = np.array([y for _, _, y, _ in cal])
    Le = logits_for(ev)
    ye = np.array([y for _, _, y, _ in ev])

    T = fit_temperature(Lc, yc, objective="ece")
    ece_before = expected_calibration_error(
        apply_temperature(Le, 1.0), ye)
    ece_after = expected_calibration_error(
        apply_temperature(Le, T), ye)
    acc = float(((apply_temperature(Le, T).argmax(1)) == ye).mean())
    print(f"T={T:.4f} acc={acc:.4f} "
          f"ece {ece_before:.4f} -> {ece_after:.4f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump({"adapter": a.adapter, "T": T, "acc": acc,
               "classes": list(J3.WORDS),
               "ece_before": ece_before, "ece_after": ece_after,
               "n_calib": len(cal), "n_eval": len(ev),
               "frozen": True, "complete": True},
              open(os.path.join(a.out, "result_glimmer_calib.json"), "w"),
              indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
