"""QLoRA fine-tune of Muse Glimmer 30B as a masked-token decision model.

Same mechanism as the 1B rung, in src/noul.py: one forward pass, two logits at
the final position, cross-entropy over just those two slots. No generation and
no new parameters beyond the LoRA adapter.

Why QLoRA and not bf16 LoRA: the base model is 59.6GB. bf16 LoRA needs the
frozen weights plus activations plus optimizer state on top of that, which is
tight on 4x40GB once activations for a 52-layer 6.7k-hidden model are included.
NF4 freezes the base at ~17GB, leaving ample room, and matches the Q4_K_M
format the model actually ships in. The 1B rung used bf16 LoRA because at 1.2B
there was nothing to save.

Adapters attach only to the TEXT tower. LoRA on the vision projections would
add parameters that never receive a gradient for a text-only task.

    python -u src/train_glimmer.py --seed 0 --epochs 1
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                          BitsAndBytesConfig)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_util import ensure_pad, checkpoint_due
from metrics import apply_temperature, fit_temperature, gate, summarize  # noqa: E402
import noul  # noqa: E402
from noul import candidates, p_fail_from_pair, target_index  # noqa: E402

# Resolved at MODULE scope: __file__ is only reliably bound at import time, and
# a helper that reads it from inside main() is an unbound-name bug the scope
# guard correctly refuses.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("JANUS_DATA", os.path.join(ROOT, "data"))
MODEL = os.environ.get("JANUS_MODEL", "meta-models/Muse-Glimmer-30B")


def load(split):
    return [json.loads(l) for l in open(f"{D}/swe_{split}.jsonl")]


def cand_ids(tok):
    """The two answer ids, asserting each is a SINGLE token.

    Glimmer's vocab has no ' yes' entry; the space-prefixed forms are 'Ġyes'
    (19690) and 'Ġno' (916). Resolving by bare word would silently pick a
    different token that never follows "Answer:".
    """
    out = {}
    for w in noul.WORDS:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"'{w}' is {len(ids)} tokens - masking impossible")
        out[w] = ids[0]
    return out



@torch.no_grad()
def score(model, tok, prompts, cids, bs=4, maxlen=192):
    model.eval()
    probs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        pair = logits[torch.arange(logits.size(0)), last][
            :, candidates(cids)].float()
        del logits
        probs.extend(p_fail_from_pair(pair.cpu().numpy(), cids).tolist())
    return np.array(probs)


def build_model(lora_r, lora_alpha):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True)
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        # text tower only: a vision adapter would never see a gradient here
        target_modules=r".*language_model\..*(q_proj|k_proj|v_proj|o_proj|"
                       r"gate_proj|up_proj|down_proj)$",
    )
    return get_peft_model(base, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--out", default=os.environ.get(
        "JANUS_OUT", os.path.join(ROOT, "out")))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cids = cand_ids(tok)
    tr, dv, te = load("train"), load("dev"), load("test")
    print(f"seed={a.seed} yes={cids['yes']} no={cids['no']} "
          f"train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)

    model = build_model(a.lora_r, a.lora_alpha)
    model.print_trainable_parameters()

    xt = [r["prompt"] for r in te]
    zy = np.array([r["label"] for r in te])
    zp = score(model, tok, xt, cids, maxlen=a.maxlen)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = int(math.ceil(len(tr) / (a.bs * a.accum) * a.epochs))
    model.train()
    t0, gi = time.time(), 0
    for s in range(steps):
        opt.zero_grad()
        last_loss = None
        for _ in range(a.accum):
            b = [tr[(gi + k) % len(tr)] for k in range(a.bs)]
            gi += a.bs
            enc = tok([r["prompt"] for r in b], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=a.maxlen).to(model.device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            pair = logits[torch.arange(len(b)), last][:, candidates(cids)]
            del logits
            tgt = torch.tensor([target_index(r["label"], cids) for r in b],
                               device=pair.device)
            last_loss = F.cross_entropy(pair.float(), tgt)
            (last_loss / a.accum).backward()
        opt.step()
        if (s + 1) % 10 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {last_loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if checkpoint_due(s + 1, steps, a.ckpt_every):
            ck = f"{a.out}/ckpt_seed{a.seed}_step{s+1}"
            model.save_pretrained(ck)
            print(f"  checkpoint -> {ck}", flush=True)

    tp = score(model, tok, xt, cids, maxlen=a.maxlen)
    dp = score(model, tok, [r["prompt"] for r in dv], cids, maxlen=a.maxlen)
    T = fit_temperature(dp, np.array([r["label"] for r in dv]))
    tpc = apply_temperature(tp, T)

    print(f"\n=== TEST n={len(te)} fail={int(zy.sum())} T={T:.2f} ===")
    for tag, p in (("zero-shot", zp), ("qlora", tp), ("qlora+T", tpc)):
        s = summarize(p, zy)
        print(f"  {tag:10s} acc {s['acc']*100:5.1f}%  brier {s['brier']:.4f}  "
              f"ece {s['ece']:.4f}  aurc {s['aurc']:.4f}  "
              f"cov@5 {s['cov5']*100:3.0f}%  cov@10 {s['cov10']*100:3.0f}%  "
              f"cov@20 {s['cov20']*100:3.0f}%", flush=True)
    print(f"  corr(p,y) on qlora = {np.corrcoef(tp, zy)[0,1]:+.4f}")
    g = gate(tpc, zy, 0.7, 0.3)
    print(f"  gate hi=.7 lo=.3: block {g['block_rate']*100:.1f}% "
          f"recall {g['fail_recall']*100:.1f}% "
          f"precision {g['block_precision']*100:.1f}%")

    res = {"model": MODEL, "seed": a.seed, "T": T, "n": int(len(zy)),
           **{t: summarize(p, zy)
              for t, p in (("zero_shot", zp), ("qlora", tp), ("qlora_T", tpc))},
           "gate_0.7_0.3": g}
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/glimmer_seed{a.seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    model.save_pretrained(f"{a.out}/glimmer_adapter_seed{a.seed}")
    print(f"\nsaved -> {a.out}/glimmer_adapter_seed{a.seed}", flush=True)


if __name__ == "__main__":
    main()
