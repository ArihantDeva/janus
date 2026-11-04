"""Eval a projection+RLCD-head artifact on cached eval features."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ("debug", "fix_input", "fix_call", "retry_wait")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--proj", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def score(X, y, head, proj):
    import torch
    id_of = {c: i for i, c in enumerate(CLASSES)}
    ref = np.array([id_of[v] for v in y])
    with torch.no_grad():
        L = (torch.tensor(X) @ proj["proj_weight"].T + proj["proj_bias"])
        L = L @ head["weight"].float().T + head["bias"].float()
    return L.argmax(1).numpy(), ref


def report(pred, ref, y, out):
    maj = max(set(y.tolist()), key=list(y).count)
    acc = float((pred == ref).mean())
    floor = float((y == maj).mean())
    per = {c: {"n": int((ref == i).sum()),
               "recall": float((pred[ref == i] == i).mean())
               if (ref == i).any() else None}
           for i, c in enumerate(CLASSES)}
    json.dump({"acc": acc, "floor": floor, "n_eval": len(y),
               "per_class": per, "method": "glimmer-frozen + proj + RLCD head"},
              open(out, "w"), indent=2)
    print(f"acc {100*acc:.1f}% (floor {100*floor:.1f}%)", flush=True)
    for c in CLASSES:
        print(f"  {c:10s} {per[c]}", flush=True)
    print(f"wrote {out}", flush=True)


def main():
    import torch
    a = build_args()
    ev = np.load(os.path.join(a.cache, "eval.npz"))
    hw = torch.load(a.head, map_location="cpu", weights_only=True)
    pr = torch.load(a.proj, map_location="cpu", weights_only=True)
    pred, ref = score(ev["X"], ev["y"], hw, pr)
    report(pred, ref, ev["y"], a.out)


if __name__ == "__main__":
    raise SystemExit(main())
