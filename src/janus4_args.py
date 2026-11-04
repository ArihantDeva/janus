"""CLI surface for the Glimmer Janus fine-tune: args only, no training logic.

Split out of train_glimmer_janus.py under the 200-LOC file rule; the trainer
imports build_args and stays focused on model, loop, and report.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("JANUS_DATA", os.path.join(ROOT, "data"))
CORPUS = os.path.join(D, "distill", "janus.jsonl")


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--resume-steps", type=int, default=0,
                      help="steps already done (only with --resume-from; "
                      "skips data already seen by continuing the cycle)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--resume-from", default="",
                      help="adapter ckpt dir to resume from (empty = fresh LoRA)")
    ap.add_argument("--eval-only", action="store_true",
                      help="score resume-from ckpt on EVAL, no training")
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--corpus", default=os.environ.get("JANUS_CORPUS", CORPUS))
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--max-rows", type=int, default=0,
                      help="cap corpus rows (shape checks only; 0 = all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out_glimmer_janus"))
    return ap.parse_args(argv)
