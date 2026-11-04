"""Train 6656->2048 projection into the FROZEN Llama RLCD head.

The deployed Llama head (weights/model_freeze0_rlcd/head.pt, 82.20%
agreement, RLCD-trained) is never modified: it is loaded frozen and only
the projection learns (cross-entropy, torch CPU). Minutes, not hours.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ("debug", "fix_input", "fix_call", "retry_wait")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    return ap.parse_args()


def load_batch(cache, head):
    import torch
    tr = np.load(os.path.join(cache, "train.npz"))
    id_of = {c: i for i, c in enumerate(CLASSES)}
    y_idx = np.array([id_of[y] for y in tr["y"]])
    hw = torch.load(head, map_location="cpu", weights_only=True)
    print(f"frozen head: {tuple(hw['weight'].shape)}", flush=True)
    return tr["X"], y_idx, hw["weight"].float(), hw["bias"].float()


def run_loop(X, y_idx, W0, b0, a):
    import torch
    import torch.nn.functional as F
    P = torch.nn.Linear(X.shape[1], W0.shape[1])
    opt = torch.optim.AdamW(P.parameters(), lr=a.lr)
    steps = int(np.ceil(len(X) / a.bs * a.epochs))
    rng = np.random.default_rng(0)
    Xt = torch.tensor(X)
    freq = np.bincount(y_idx, minlength=4)
    cw = torch.tensor(freq.sum() / (4 * freq), dtype=torch.float32)
    for s in range(steps):
        idx = rng.integers(0, len(Xt), a.bs)
        logits = (Xt[idx] @ P.weight.T + P.bias) @ W0.T + b0
        loss = F.cross_entropy(logits, torch.tensor(y_idx[idx]), weight=cw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 50 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.4f}", flush=True)
    return P


def save_proj(P, out):
    import torch
    os.makedirs(out, exist_ok=True)
    torch.save({"proj_weight": P.weight.detach(),
                "proj_bias": P.bias.detach()},
               os.path.join(out, "proj.pt"))
    with open(os.path.join(out, "METHOD.md"), "w") as fh:
        fh.write("# Glimmer RLCD (projection method)\n\n"
                 "Frozen Glimmer 30B encoder (bidirectional patch) -> "
                 "trained linear 6656->2048 -> FROZEN deployed Llama RLCD "
                 "head (weights/model_freeze0_rlcd/head.pt, 82.20%). "
                 "Only the projection learned; every decision weight is "
                 "RLCD-trained. Corpus: group-disjoint janus.jsonl split.\n")
    print(f"wrote {out}/proj.pt", flush=True)


def main():
    a = build_args()
    X, y_idx, W0, b0 = load_batch(a.cache, a.head)
    P = run_loop(X, y_idx, W0, b0, a)
    save_proj(P, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
