"""Glimmer 30B frozen-encoder + head-only distill (winning Llama recipe).

freeze0 method, ported: frozen encoder, only a linear head trains,
distill-only, 6 epochs, head-lr 3e-4. Mechanical differences: 4-bit NF4
base, language_model submodule, trust_remote_code, generic causal-flag
patch. 4-class Janus task. Group-disjoint split asserted in code.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.environ.get("JANUS_CORPUS",
                         os.path.join(ROOT, "data", "distill", "janus.jsonl"))
MODEL = os.environ.get("JANUS_MODEL", "meta-models/Muse-Glimmer-30B")

CLASSES = ("debug", "fix_input", "fix_call", "retry_wait")
_CAUSAL_ATTRS = ("is_causal", "causal", "_is_causal", "causal_mask")

from glimmer_distill_data import (  # noqa: E402
    encode_input, load_corpus, split_by_group, teacher_target)


def count_causal(model):
    n = 0
    for _, mod in model.named_modules():
        for attr in _CAUSAL_ATTRS:
            try:
                if getattr(mod, attr, None) is True:
                    n += 1
                    break
            except Exception:
                continue
    return n


def patch_bidirectional(model):
    n = 0
    for _, mod in model.named_modules():
        for attr in _CAUSAL_ATTRS:
            try:
                if getattr(mod, attr, None) is True:
                    setattr(mod, attr, False)
                    n += 1
                    break
            except Exception:
                continue
    return n


def build_model(dev):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForImageTextToText, AutoTokenizer, \
        BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "right"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    full = AutoModelForImageTextToText.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True)
    from train_glimmer_freeze_head import GlimmerClf
    enc = full.model.language_model
    enc.gradient_checkpointing_enable()
    enc.config.use_cache = False
    n = patch_bidirectional(enc)
    if count_causal(enc):
        raise RuntimeError(f"{count_causal(enc)} causal flags survived")
    print(f"patched {n}; frozen encoder: only the head trains", flush=True)
    model = GlimmerClf(enc, enc.config.hidden_size,
                       len(CLASSES)).to(dev)
    for p in model.encoder.parameters():
        p.requires_grad = False
    return model, tok


def run_distill(model, tok, tr, encode, a, dev):
    import torch
    import torch.nn.functional as F
    Ttr = np.stack([teacher_target(r) for r in tr])
    steps = int(math.ceil(len(tr) / a.bs * a.epochs))
    opt = torch.optim.AdamW(model.head.parameters(), lr=a.head_lr,
                            weight_decay=0.01)
    rng, oi, t0 = np.random.default_rng(a.seed), 0, time.time()
    order = rng.permutation(len(tr))
    model.train()
    for s in range(steps):
        if oi + a.bs > len(order):
            order, oi = rng.permutation(len(tr)), 0
        idx = order[oi:oi + a.bs].tolist()
        oi += a.bs
        b = {k: v.to(dev) for k, v in encode([tr[i] for i in idx]).items()}
        soft = torch.softmax(torch.tensor(
            Ttr[idx] / a.temp, dtype=torch.float32), -1).to(dev)
        loss = F.kl_div(F.log_softmax(model(**b).float(), -1), soft,
                        reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        opt.step()
        if (s + 1) % 50 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.4f}", flush=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {s+1}")


def evaluate(model, tok, ev, encode, dev):
    import torch
    import numpy as _np
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(ev), 8):
            b = {k: v.to(dev) for k, v in encode(ev[i:i+8]).items()}
            outs.append(torch.softmax(model(**b).float(), -1).cpu().numpy())
    P = _np.concatenate(outs, 0) if outs else _np.zeros((0, 4))
    return P.argmax(1), _np.array([CLASSES.index(r["_label"]) for r in ev])


def report(model, pred, ev, tr, a):
    import torch
    y = [CLASSES.index(r["_label"]) for r in ev]
    y = np.array(y)
    maj = collections.Counter(r["_label"] for r in tr).most_common(1)[0][0]
    floor = sum(r["_label"] == maj for r in ev) / len(ev)
    acc = float((pred == y).mean())
    print(f"eval acc {100*acc:.1f}% (floor {100*floor:.1f}%)", flush=True)
    for i, c in enumerate(CLASSES):
        m = y == i
        if m.any():
            print(f"  {c:10s} recall {100*float((pred[m]==i).mean()):.1f}%",
                  flush=True)
    torch.save(model.head.state_dict(), os.path.join(a.out, "head.pt"))
    json.dump({"seed": a.seed, "acc": acc, "floor": floor,
               "frozen": True, "n_train": len(tr), "n_eval": len(ev)},
              open(os.path.join(a.out, "result_glimmer_freeze.json"), "w"),
              indent=2)


def load_splits(a):
    rows = load_corpus()
    if a.max_rows:
        rows = rows[:a.max_rows]
    tr_all, ev, _ = split_by_group(rows, seed=a.seed)
    tra, cal, _ = split_by_group(tr_all, frac=0.12, seed=a.seed + 500)
    if a.shard:
        k, n = a.shard
        tra = [r for i, r in enumerate(tra) if i % n == k]
        print(f"shard {k}/{n}: train {len(tra)} rows", flush=True)
    print(f"train {len(tra)} / calib {len(cal)} / eval {len(ev)}",
          flush=True)
    return tra, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--shard", nargs=2, type=int, default=None,
                    metavar=("K", "N"),
                    help="train on stride K of N (data-parallel head "
                    "shards; average heads afterwards)")
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, ev = load_splits(a)
    model, tok = build_model(dev)

    def encode(rs):
        return tok([encode_input(r) for r in rs], return_tensors="pt",
                   padding=True, truncation=True, max_length=a.maxlen)

    run_distill(model, tok, tr, encode, a, dev)
    pred, _ = evaluate(model, tok, ev, encode, dev)
    report(model, pred, ev, tr, a)


if __name__ == "__main__":
    raise SystemExit(main())
