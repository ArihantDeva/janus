"""Average data-parallel head shards into one deployable head."""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    states = [torch.load(os.path.join(s, "head.pt"), map_location="cpu",
                         weights_only=True) for s in a.shards]
    keys = states[0].keys()
    assert all(list(s.keys()) == list(keys) for s in states), "head mismatch"
    avg = {k: sum(s[k].float() for s in states) / len(states) for k in keys}
    torch.save(avg, os.path.join(a.out, "head.pt"))
    print(f"averaged {len(states)} shards -> {a.out}/head.pt")


if __name__ == "__main__":
    raise SystemExit(main())
