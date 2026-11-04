"""Cache frozen Glimmer pooled features + labels to npz.

One GPU pass; everything downstream (projection training, eval) runs on
cached features in minutes. Subset-capable for time-boxing.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glimmer_distill_data import encode_input, load_corpus, split_by_group


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--max-eval", type=int, default=1000)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_encoder():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, \
        BitsAndBytesConfig
    from train_glimmer_freeze import MODEL, patch_bidirectional, count_causal
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
    enc = full.model.language_model
    enc.config.use_cache = False
    patch_bidirectional(enc)
    assert count_causal(enc) == 0, "causal flags survived"
    for p in enc.parameters():
        p.requires_grad = False
    return enc.eval(), tok


def dump_split(enc, tok, rs, out, name, maxlen):
    import torch
    from glimmer_distill_data import teacher_target
    outs = []
    with torch.no_grad():
        for i in range(0, len(rs), 8):
            e = tok([encode_input(r) for r in rs[i:i+8]],
                    return_tensors="pt", padding=True,
                    truncation=True, max_length=maxlen).to("cuda")
            h = enc(**e).last_hidden_state.float()
            m = e["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * m).sum(1) / m.sum(1).clamp_min(1e-6)).cpu())
            if (i // 8 + 1) % 25 == 0:
                print(f"  {min(i+8, len(rs))}/{len(rs)}", flush=True)
    X = torch.cat(outs, 0).numpy()
    d = {"X": X, "y": np.array([r["_label"] for r in rs]),
         "T": np.stack([teacher_target(r) for r in rs])}
    np.savez_compressed(os.path.join(out, f"{name}.npz"), **d)
    print(f"wrote {name}: {d['X'].shape}", flush=True)


def main():
    a = build_args()
    enc, tok = load_encoder()
    rows = load_corpus()
    tr_all, ev, _ = split_by_group(rows, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    dump_split(enc, tok, tr_all[:a.max_train], a.out, "train", a.maxlen)
    dump_split(enc, tok, ev[:a.max_eval], a.out, "eval", a.maxlen)


if __name__ == "__main__":
    raise SystemExit(main())
