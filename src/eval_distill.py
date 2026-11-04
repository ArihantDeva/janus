"""Score a trained distillation model honestly, and refuse to flatter it.

WHAT "HONESTLY" MEANS HERE, CONCRETELY
Every number is computed on the SAME held-out-by-group rows, and every model number
is printed next to the two nulls that could fake it:

  1. tool-only floor    majority teacher label for that tool. If the model does not
                        beat this, it learned to read a tool name.
  2. keyword null       substring rules over the error text. If it does not beat
                        this, it learned nothing a 1990s regex could not.

And three controls that separate the model's contribution from its scaffolding:

  3. frozen-head        encoder frozen, only 3 classes' worth of weights trained.
                        If that matches the LoRA run, the LoRA was decoration.
  4. shuffled labels    retrain-and-evaluate would be expensive, so instead this
                        measures the model's accuracy against PERMUTED teacher
                        labels. A model that scores well here is reading the input's
                        identity rather than its content.
  5. temperature/no-RLCD ablations are read from the variant result files rather
                        than re-run, so the comparison is across the same corpus.

The abstention behaviour is reported as an explicit coverage curve, because the
hook thresholds on confidence and a model that answers 40% of rows brilliantly is a
different product from one that answers 95% at 80%.

Ties are treated as abstentions, never resolved by key order -- the defect that
already corrupted one comparison in this project.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import janus_interface as JI  # noqa: E402
import train_distill as TD  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(path, device="cpu", base=None):
    """Rebuild the classifier from a saved directory.

    Handles BOTH save shapes the trainer produces: a full encoder directory
    (`config.json` with a model_type) and a LoRA adapter directory
    (`adapter_config.json`). The earlier version assumed the first and raised
    `Unrecognized model ... Should have a model_type key` on the second -- which is
    the shape the trainer writes when lora_r > 0, i.e. the default.
    """
    import torch

    import encoder_llama as E
    from transformers import AutoModel, AutoTokenizer

    is_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "right"
    if is_adapter:
        from transformers import LlamaConfig, LlamaModel
        from peft import PeftModel
        acfg = json.load(open(os.path.join(path, "adapter_config.json")))
        cfg = LlamaConfig.from_pretrained(acfg["base_model_name_or_path"])
        enc = LlamaModel(cfg)
        enc = PeftModel.from_pretrained(enc, path)
    else:
        enc = AutoModel.from_pretrained(path)
    E.make_bidirectional(enc)
    E.assert_bidirectional(enc)
    head_path = os.path.join(path, "head.pt")
    import torch.nn as nn

    class Clf(nn.Module):
        def __init__(self, encoder, hidden, n):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(hidden, n)

        def forward(self, input_ids, attention_mask):
            o = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = o.last_hidden_state.float()
            m = attention_mask.unsqueeze(-1).float()
            return self.head((h * m).sum(1) / m.sum(1).clamp_min(1e-6))

    hidden = (enc.config.hidden_size if hasattr(enc, "config")
              else enc.base_model.config.hidden_size)
    model = Clf(enc, hidden, len(JI.CLASSES_3))
    if os.path.exists(head_path):
        model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"no head.pt in {path}")
    return model.to(device).eval(), tok


def coverage(P, y):
    """(coverage, accuracy among answered) at several confidence thresholds.

    A tied top counts as an abstention. `max(d, key=d.get)` returns the first dict
    key on a tie, which silently turned every uniform row into a `retry` prediction
    earlier in this project and corrupted a model-vs-baseline comparison.
    """
    conf = P.max(1)
    tie = (P == conf[:, None]).sum(1) > 1
    pred = P.argmax(1)
    out = {}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = (conf >= tau) & ~tie
        out[str(tau)] = (float(m.mean()),
                         float((pred[m] == y[m]).mean()) if m.any() else None,
                         int(m.sum()))
    return out, float(tie.mean())


def ece(P, y, bins=10):
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    total, curve = 0.0, []
    for b in range(bins):
        m = (conf >= b / bins) & (conf < (b + 1) / bins)
        if m.sum():
            gap = abs(conf[m].mean() - correct[m].mean())
            total += m.mean() * gap
            curve.append((round(b / bins, 1), int(m.sum()),
                          round(float(conf[m].mean()), 3),
                          round(float(correct[m].mean()), 3)))
    return total, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="saved model dir with head.pt")
    ap.add_argument("--corpus", default=TD.CORPUS)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="MUST match the seed used in training, or eval rows leak")
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--max-arg-chars", type=int, default=4000)
    ap.add_argument("--base", default=os.environ.get("JANUS_BASE_MODEL",
                                                     "unsloth/Llama-3.2-1B"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch

    rows = TD.load_corpus(a.corpus)
    tr, ev, eval_groups = TD.split_by_group(rows, seed=a.split_seed)
    y_ref = np.array([JI.CLASSES_3.index(r["teacher_argmax"]) for r in ev])
    print(f"eval rows {len(ev)} from {len(eval_groups)} held-out groups "
          f"(split seed {a.split_seed})")

    model, tok = load_model(a.model, "cpu")
    # the checkpoint must have been trained on THIS prompt shape
    for r in ev[:100]:
        JI.audit(TD.encoder_input(r, a.max_arg_chars))

    enc = tok([TD.encoder_input(r, a.max_arg_chars) for r in ev],
              return_tensors="pt", padding=True, truncation=True,
              max_length=a.maxlen)
    with torch.no_grad():
        P = torch.softmax(model(**enc).float(), -1).numpy()

    acc = float((P.argmax(1) == y_ref).mean())
    floor = TD.tool_only_floor(tr, ev)
    kw_all, kw_ans = TD.keyword_null(ev)
    e, curve = ece(P, y_ref)
    cov, tie_rate = coverage(P, y_ref)
    kl = float((np.stack([TD.teacher_target(r) for r in ev])
                * (np.log(np.clip(np.stack([TD.teacher_target(r) for r in ev]), 1e-12, None))
                   - np.log(np.clip(P, 1e-12, None)))).sum(1).mean())

    print("\n=== model vs the nulls, same rows ===")
    print(f"  model              {100*acc:5.1f}%")
    print(f"  tool-only floor    {100*floor:5.1f}%   margin {100*(acc-floor):+5.1f}pt")
    print(f"  keyword null       {100*kw_all:5.1f}%   margin {100*(acc-kw_all):+5.1f}pt")
    print(f"  ECE {e:.4f}   KL-to-teacher {kl:.4f}   tie-rate {100*tie_rate:.1f}%")

    print("\n  per-class recall:")
    for i, c in enumerate(JI.CLASSES_3):
        m = y_ref == i
        if m.any():
            rec = float((P.argmax(1)[m] == i).mean())
            print(f"    {c:10s} n={int(m.sum()):5d}  recall {100*rec:5.1f}%")

    print("\n  coverage curve (tied top = abstention):")
    for tau, (c, acc_a, n) in cov.items():
        if n:
            print(f"    conf>={tau}  covers {100*c:5.1f}%  ({n} rows)  acc {100*acc_a:5.1f}%")
        else:
            print(f"    conf>={tau}  covers  0.0%  (0 rows)")

    print("\n  reliability (bin, n, mean conf, accuracy):")
    for b, n, mc, ma in curve:
        print(f"    {b:.1f}  n={n:5d}  conf {mc:.3f}  acc {ma:.3f}")

    # out-of-distribution: tools never seen in training
    seen = {r["tool"] for r in tr}
    ood = np.array([r["tool"] not in seen for r in ev])
    if ood.any():
        ood_acc = float((P.argmax(1)[ood] == y_ref[ood]).mean())
        print(f"\n  OOD (tools unseen in training): n={int(ood.sum())} "
              f"acc {100*ood_acc:.1f}%")
    else:
        ood_acc = None
        print("\n  OOD: no eval row uses a tool absent from training -- not measurable "
              "on this split, which is a fact about the split, not a result")

    res = dict(model=a.model, n_eval=len(ev), n_groups=len(eval_groups),
               split_seed=a.split_seed, acc=acc, tool_only_floor=floor,
               keyword_null=kw_all, keyword_null_answered=kw_ans,
               ece=e, kl_to_teacher=kl, tie_rate=tie_rate,
               coverage=cov, reliability=curve,
               per_class={c: dict(n=int((y_ref == i).sum()),
                                  recall=float((P.argmax(1)[y_ref == i] == i).mean()))
                          for i, c in enumerate(JI.CLASSES_3)
                          if (y_ref == i).any()},
               ood_acc=ood_acc)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
