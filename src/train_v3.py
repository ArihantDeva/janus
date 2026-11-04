"""Train Llama-3.2-1B on state + question + choices -> calibrated response class.

WHAT THE MODEL SEES
Only what `janus_interface` renders: the failed call's state (tool, arguments,
error), one question, and the three choice criteria. No instructions, no role
text, no examples. `janus_interface.audit` runs before training so the constraint
cannot be violated by accident.

THE READ-OUT
Letter-code choice (nimble's mechanism): each choice is one token at the answer
boundary, and the distribution is a softmax over exactly those three logits. The
model cannot emit anything outside the choice set, so there is no parsing step and
no invalid answer, and the probabilities are directly usable as confidence.

WHY THE CONTEXT LADDER
Measured on 16,679 real failures (`measure_context.py`): with ~700 tokens of prompt
overhead, 2048 covers 95.0% of failures and 8192 covers 100%; the longest observed
is 6,010 tokens. So stage 1 trains at 2048 (cheap, nearly all the data) and stage 2
continues at 8192 for the tail. Training at 32K would spend three quarters of every
sequence on padding for no extra coverage.

WHAT IS REPORTED BESIDES ACCURACY
  tool-only floor   the accuracy a lookup table on `tool` alone reaches. A claim
                    must be a margin over this, or it is a claim that the model
                    learned to read a tool name.
  ECE, Brier        calibration -- the actual product claim, because the hook
                    thresholds on probability.
  coverage@tau      the operating curve: how much is answered at a confidence bar,
                    and how accurate those answers are.
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
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import janus_interface as JI  # noqa: E402
from choice_codes import code_ids  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, split_dir="responses_v3"):
    p = os.path.join(ROOT, "data", split_dir, f"{name}.jsonl")
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if r["y"] in JI.CLASSES_3]


def prompt_for(row, max_arg_chars=8000):
    """One row -> model input. Goes through the single owner; never built here."""
    st = JI.build_state(tool=row.get("tool"), arguments=row.get("arguments"),
                        error=row.get("error"))
    return JI.render(st, max_arg_chars=max_arg_chars)


def option_texts():
    return [JI.CRITERIA[c] for c in JI.CLASSES_3]


def tool_only_floor(train, held):
    """Accuracy of "guess the majority label for this tool"."""
    table = collections.defaultdict(collections.Counter)
    for r in train:
        table[r.get("tool")][r["y"]] += 1
    best = {t: c.most_common(1)[0][0] for t, c in table.items()}
    default = collections.Counter(r["y"] for r in train).most_common(1)[0][0]
    return sum(1 for r in held if best.get(r.get("tool"), default) == r["y"]) / max(1, len(held))


def metrics(P, y):
    pred = P.argmax(1)
    acc = float((pred == y).mean())
    oh = np.eye(len(JI.CLASSES_3))[y]
    brier = float(((P - oh) ** 2).sum(1).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())
    conf = P.max(1)
    correct = (pred == y).astype(float)
    ece = 0.0
    for b in range(10):
        m = (conf >= b / 10) & (conf < (b + 1) / 10)
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    cov = {}
    for tau in (0.5, 0.7, 0.9):
        m = conf >= tau
        cov[tau] = (float(m.mean()),
                    float(correct[m].mean()) if m.any() else float("nan"))
    per = {}
    for k, c in enumerate(JI.CLASSES_3):
        m = y == k
        if m.sum():
            per[c] = (int(m.sum()), float((pred[m] == k).mean()))
    return dict(acc=acc, brier=brier, nll=nll, ece=ece,
                cov={str(k): v for k, v in cov.items()}, per=per)


@torch.no_grad()
def predict(model, tok, rows, maxlen, bs=4, max_arg_chars=8000):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ps = [prompt_for(r, max_arg_chars) for r in chunk]
        cd = code_ids(tok, ps[0], option_texts())
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True,
                  max_length=maxlen).to(model.device)
        o = model(**enc)
        last = enc["attention_mask"].sum(1) - 1
        lg = o.logits[torch.arange(len(chunk)), last][:, cd]
        del o
        out.append(torch.softmax(lg.float(), -1).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("JANUS_BASE_MODEL",
                                                      "unsloth/Llama-3.2-1B"))
    ap.add_argument("--stages", default="2048,8192")
    ap.add_argument("--epochs", type=float, default=3.0, help="per stage")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--min-class", type=int, default=100)
    ap.add_argument("--max-arg-chars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out_v3"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = load("train")
    held = load("rule_held")
    counts = collections.Counter(r["y"] for r in train)
    below = sorted(c for c, n in counts.items() if n < a.min_class)
    if below:
        # The choice letters are baked into the read-out, so a silently smaller
        # class set would relabel every prediction. Refuse rather than shrink.
        raise SystemExit(
            f"classes below --min-class {a.min_class}: {below}. The choice letters "
            f"are part of the read-out; change janus_interface.CLASSES_3 deliberately.")

    y_tr = np.array([JI.choice_index(r["y"]) for r in train])
    y_he = np.array([JI.choice_index(r["y"]) for r in held])
    floor = tool_only_floor(train, held)
    print(f"train {len(train)}  held {len(held)}  classes {JI.CLASSES_3}")
    print(f"tool-only floor on held: {100*floor:.1f}%  <- the margin that matters")
    print(f"balance: {dict(collections.Counter(r['y'] for r in train).most_common())}")

    for r in train[:200]:
        JI.audit(prompt_for(r, a.max_arg_chars))
    print("interface audit: ok (state+question+choices only)")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # keep the FRONT: the error is the last state field, so right-truncation drops
    # the tail of a long argument blob rather than the error itself
    tok.truncation_side = "right"

    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    model = get_peft_model(base, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    cw = torch.tensor([math.sqrt(len(train) / (len(JI.CLASSES_3) * counts[c]))
                       for c in JI.CLASSES_3], dtype=torch.float32, device=dev)
    print(f"class weights: {[f'{c}={w:.2f}' for c, w in zip(JI.CLASSES_3, cw.tolist())]}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    stages = [int(s) for s in a.stages.split(",") if s.strip()]
    rng = np.random.default_rng(a.seed)
    results = {}

    for stage_i, maxlen in enumerate(stages, 1):
        enc = tok([prompt_for(r, a.max_arg_chars) for r in train],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=maxlen)
        print(f"\n=== stage {stage_i}/{len(stages)} maxlen={maxlen} ===")
        cd = code_ids(tok, prompt_for(train[0], a.max_arg_chars), option_texts())
        print(f"code ids at boundary: {cd}")

        steps = int(math.ceil(len(train) / a.bs * a.epochs))
        order, oi, t0 = rng.permutation(len(train)), 0, time.time()
        model.train()
        for s in range(steps):
            if oi + a.bs > len(order):
                order, oi = rng.permutation(len(train)), 0
            idx = order[oi:oi + a.bs].tolist()
            oi += a.bs
            batch = {k: v[idx].to(dev) for k, v in enc.items()}
            out = model(**batch)
            last = batch["attention_mask"].sum(1) - 1
            logits = out.logits[torch.arange(len(idx)), last][:, cd]
            del out
            loss = F.cross_entropy(logits.float(),
                                   torch.tensor(y_tr[idx], device=dev), weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (s + 1) % 200 == 0 or s == 0:
                print(f"  step {s+1}/{steps} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        del enc
        if dev == "cuda":
            torch.cuda.empty_cache()

        P = predict(model, tok, held, maxlen, a.max_arg_chars)
        m = metrics(P, y_he)
        results[f"stage{stage_i}_maxlen{maxlen}"] = m
        print(f"  held acc {100*m['acc']:5.1f}%  (floor {100*floor:.1f}%, "
              f"margin {100*(m['acc']-floor):+5.1f}pt)  ece {m['ece']:.4f}  "
              f"brier {m['brier']:.4f}")
        print(f"  coverage@0.7 {100*m['cov']['0.7'][0]:5.1f}% at acc "
              f"{100*m['cov']['0.7'][1]:5.1f}%")
        print("  per-class: " + "  ".join(
            f"{k}={100*v[1]:.0f}%(n{v[0]})" for k, v in m["per"].items()))

    out = {"seed": a.seed, "stages": stages, "n_train": len(train),
           "n_held": len(held), "tool_only_floor": floor, "lr": a.lr,
           "classes": list(JI.CLASSES_3), "results": results}
    json.dump(out, open(f"{a.out}/result_seed{a.seed}.json", "w"), indent=2)
    model.save_pretrained(f"{a.out}/adapter_seed{a.seed}")
    tok.save_pretrained(f"{a.out}/adapter_seed{a.seed}")
    print(f"\nwrote {a.out}/result_seed{a.seed}.json and adapter")


if __name__ == "__main__":
    main()
