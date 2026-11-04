"""4-slot softmax scoring for the Glimmer Janus verdict task.

Split out of train_glimmer_janus.py under the 200-LOC file rule: no
generation, one forward pass, read out the four verdict words' logits.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

try:
    import torch
except ImportError:  # cluster-only deps; CPU suites import build_args only
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import janus3 as J3  # noqa: E402
import janus4_heartbeat as JH  # noqa: E402


def _no_grad():
    """torch.no_grad on the cluster, identity decorator on CPU import."""
    if torch is None:
        def identity(fn):
            return fn
        return identity
    return torch.no_grad()


@_no_grad()
def score(model, tok, prompts, cids, bs=2, maxlen=512):
    """4-slot softmax over the verdict words. No generation."""
    model.eval()
    slots = J3.candidates(cids)
    out = []
    t0 = time.time()
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        quad = logits[torch.arange(logits.size(0)), last][:, slots].float()
        del logits
        out.append(quad.softmax(-1).cpu().numpy())
        JH.progress("score", i + bs, len(prompts), t0, every=50,
                    step=i // bs + 1)
    return np.concatenate(out)
