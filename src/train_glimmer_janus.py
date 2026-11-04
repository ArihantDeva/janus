"""Glimmer 30B QLoRA trainer for the Janus 4-verdict task.

Same masked-token mechanism as train_glimmer.py (no new head: one forward
pass, read out the four verdict words' logits), but the TASK is Janus, not
p(fail): post-failure rows, four verdict words from janus3, targets from the
teacher where it speaks and from the retry_wait rule where it is silent.

TARGET HONESTY
Integer labels per row (teacher argmax, retry_wait by rule); class weights
counter the 48/23/25/4 imbalance.

SPLIT
By `group` (project+session), disjoint by assertion.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, \
        prepare_model_for_kbit_training
    from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                              BitsAndBytesConfig)
except ImportError:  # cluster-only deps; CPU suites import build_args only
    torch, F = None, None
    LoraConfig = get_peft_model = prepare_model_for_kbit_training = None
    AutoModelForImageTextToText = AutoTokenizer = BitsAndBytesConfig = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import janus3 as J3  # noqa: E402
import janus4_args as JA  # noqa: E402
import janus4_data as JD  # noqa: E402
import janus4_heartbeat as JH  # noqa: E402
import janus4_score as JS  # noqa: E402
from train_util import checkpoint_due, steps_for  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("JANUS_DATA", os.path.join(ROOT, "data"))
MODEL = os.environ.get("JANUS_MODEL", "meta-models/Muse-Glimmer-30B")


def cand_ids(tok):
    """The four read-out LETTERS, each asserting a SINGLE token.

    Words failed: Glimmer splits fix_input/fix_call/retry_wait into two
    tokens each (measured 2026-11-01). Letters A-D are guaranteed single.
    A multi-token letter fails loudly instead of scoring the wrong slot.
    """
    out = {}
    for i in range(len(J3.WORDS)):
        c = J3.CODES[i]
        ids = tok.encode(" " + c, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"'{c}' is {len(ids)} tokens - masking impossible")
        out[c] = ids[0]
    return out


def bnb_cfg():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)


def build_model(lora_r, lora_alpha, resume_from=None):
    if resume_from:
        from peft import PeftModel
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL, quantization_config=bnb_cfg(),
            device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, resume_from,
                                          is_trainable=True)
        # fresh-path memory diet: no checkpointing OOMs eff-batch 16+ (Nov02)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        return model
    bnb = bnb_cfg()
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True)
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True)
    base.enable_input_require_grads()
    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model\..*(q_proj|k_proj|v_proj|o_proj|"
                       r"gate_proj|up_proj|down_proj)$",
    )
    return get_peft_model(base, cfg)


def train_loop(model, tok, tr, cids, a):
    """QLoRA pass over hard integer labels with class weights."""
    slots = J3.candidates(cids)
    lab = torch.tensor(np.array([y for _, _, y, _ in tr]), dtype=torch.long)
    cw = torch.tensor(JD.class_weights(
        np.bincount(lab.numpy(), minlength=4)), dtype=torch.float32)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = steps_for(len(tr), a.bs, a.accum, a.epochs)
    model.train()
    t0, gi = time.time(), 0
    if a.resume_from and a.resume_steps > 0:
        # continue the data cycle where the ckpt left off instead of
        # replaying the head (replay re-memorizes early rows and spikes)
        gi = (a.resume_steps * a.bs * a.accum) % len(tr)
    for s in range(steps):
        opt.zero_grad()
        for _ in range(a.accum):
            b = [(gi + k) % len(tr) for k in range(a.bs)]
            gi += a.bs
            enc = tok([tr[j][0] for j in b], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=a.maxlen).to(model.device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            quad = logits[torch.arange(len(b)), last][:, slots].float()
            del logits
            tb = lab[b].to(quad.device)
            loss = F.cross_entropy(quad, tb, weight=cw.to(quad.device))
            loss = loss / a.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], a.grad_clip)
        opt.step()
        JH.progress("step", s + 1, steps, t0, every=10)
        if (s + 1) % 10 == 0 or s == 0:
            print(f"  loss {loss.item()*a.accum:.4f}", flush=True)
        if checkpoint_due(s + 1, steps, a.ckpt_every):
            ck = f"{a.out}/janus_ckpt_seed{a.seed}_step{s+1}"
            model.save_pretrained(ck)
            print(f"  checkpoint -> {ck}", flush=True)
    return steps


def report(tag, P, y):
    m = JD.metrics4(P, y)
    rec = " ".join(f"{c}={v[1]*100:.0f}%(n={v[0]})" for c, v in m["per"].items())
    print(f"  {tag:10s} acc {m['acc']*100:5.1f}%  ece {m['ece']:.4f}  {rec}",
          flush=True)
    return m


def main(argv=None):
    a = JA.build_args(argv)
    JH.heartbeat("janus-glimmer")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cids = cand_ids(tok)
    print(f"seed={a.seed} " + " ".join(
        f"{J3.CODES[i]}={cids[J3.CODES[i]]}({w})"
        for i, w in enumerate(J3.WORDS)), flush=True)
    rows = JD.load_rows(a.corpus)
    if a.max_rows > 0:
        # shuffle before slicing: raw head is label-skewed (retry_wait n=2
        # in the 500-slice) and the skew is uninformative, not a signal
        rng = np.random.default_rng(a.seed)
        rows = [rows[i] for i in rng.permutation(len(rows))[:a.max_rows]]
    tr, ev = JD.split_groups(rows, a.eval_frac, a.seed)
    print(f"train={len(tr)} eval={len(ev)} "
          f"balance={JD.balance(tr)}", flush=True)
    print("[janus-glimmer] loading base model...", flush=True)
    model = build_model(a.lora_r, a.lora_alpha,
                        resume_from=a.resume_from or None)
    model.print_trainable_parameters()
    zp = JS.score(model, tok, [p for p, _, _, _ in ev], cids, maxlen=a.maxlen)
    zy = np.array([y for _, _, y, _ in ev])
    if a.eval_only:
        print(f"\n=== EVAL n={len(zy)} (eval-only {a.resume_from}) ===")
        res = {"model": MODEL, "seed": a.seed,
               "zero_shot": None,
               "qlora": report("qlora", zp, zy)}
        JD.save_result(a.out, a.seed, res)
        print(f"\nsaved -> {a.out}", flush=True)
        return
    train_loop(model, tok, tr, cids, a)
    tp = JS.score(model, tok, [p for p, _, _, _ in ev], cids, maxlen=a.maxlen)
    print(f"\n=== EVAL n={len(zy)} ===")
    res = {"model": MODEL, "seed": a.seed,
           "zero_shot": report("zero-shot", zp, zy),
           "qlora": report("qlora", tp, zy)}
    JD.save_result(a.out, a.seed, res)
    model.save_pretrained(f"{a.out}/glimmer_janus_adapter_seed{a.seed}")
    print(f"\nsaved -> {a.out}/glimmer_janus_adapter_seed{a.seed}", flush=True)


if __name__ == "__main__":
    main()
