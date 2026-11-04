"""Turn Llama-3.2-1B into a Janus-type model.

This is the whole thing, in one file, combining the two mechanisms that were
verified against real implementations:

READ-OUT  (from bespokelabsai/nimble)
    Each candidate answer gets a ONE-TOKEN LETTER CODE (A, B, C, ...) resolved at
    the prompt's answer boundary. The distribution is a softmax over exactly
    those K code logits. Nothing else is representable, so the model cannot emit
    an invalid answer and there is no parsing step.
    Verified on Llama-3.2-1B: all 26 letters are single tokens at a clean
    boundary; the encoding must be stripped-prefix + one space + code, because
    keeping a dangling space makes BPE replace it rather than append.

OBJECTIVE  (nimble's, unchanged)
    Cross-entropy over exactly the K candidate code logits, with sqrt class
    weighting for the rare classes. That is the whole loss. No length
    normalisation, no auxiliary term, no margin -- deliberately, because the
    claim being tested is "nimble's recipe works on this base model", and any
    extra term would make the result unattributable.

    I started with laya/open-janus's RLCD loss (soft NLL + Brier + differentiable
    ECE, which exists to fix overconfidence from one-hot targets) and removed it.
    Two reasons, both concrete: nimble gets 90.1% on a 324-item ladder with plain
    CE, so the calibration machinery is not required to reach the working point;
    and this repo has already measured that ECE here is fixable post-hoc, so
    optimising it in-loop buys nothing that survives. Soft targets would also
    have required inventing the ambiguity weights, and run 33 showed what happens
    when a hand-written rule stands in for ground truth.

WHY THE READ-OUT TOGETHER WITH THE LABELS
    Runs 29-31 changed read-out without changing data and moved in-distribution
    accuracy to 89.3% while benchmarks stayed at the floor. Run 33 then showed
    the labels themselves were the ceiling (accuracy equalled agreement with the
    label rule). So the read-out was never the only problem. What the two source
    repos supply between them is the complete recipe: nimble's code read-out and
    its CE objective, applied to labels that come from real observed failures.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No benchmark. No R-Judge, no ASB, no length control, no threshold tuning. The
user asked for one thing: turn Llama into a Janus-type model. Success is measured
on held-out real data by agreement with a reference distribution and by
calibration, not by a number engineered to look good.
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

from choice_codes import CODES, code_ids  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# THE TASK, FIXED
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "You are diagnosing a failed tool call made by a software agent.\n"
    "Read the failure and decide what the agent should do next.\n"
    "Answer with the single letter of the best option."
)

#: The answer set. Order is the code order (A, B, C, ...) and must never change
#: without retraining, because the code letters are what the model learns.
CLASSES = (
    "retry",       # A - environment was slow; re-run unchanged
    "reauth",      # B - credentials wrong; retrying is futile
    "fix_call",    # C - tool rejected the ARGUMENTS; correct the call
    "fix_input",   # D - path/handle missing; correct the input
    "debug",       # E - action itself broken; read the error, change the code
    "abort",       # F - interrupted; stop
)

OPTION_TEXT = {
    "retry": "the environment was slow, so run the same call again unchanged",
    "reauth": "the credentials are wrong, so re-authenticate before retrying",
    "fix_call": "the tool rejected the arguments, so correct the call",
    "fix_input": "the path or handle does not exist, so correct the input",
    "debug": "the action itself is broken, so read the error and change the code",
    "abort": "the user interrupted, so stop and do not retry",
}


def build_prompt(text, classes=CLASSES):
    lines = [f"{CODES[i]}) {OPTION_TEXT[c]}" for i, c in enumerate(classes)]
    return (f"{INSTRUCTION}\n\nFailure:\n{text.strip()}\n\n"
            f"Options:\n" + "\n".join(lines) + "\n\nAnswer: ")


# ---------------------------------------------------------------------------
# OBJECTIVE: RLCD-shaped
# ---------------------------------------------------------------------------

def nimble_loss(logits, y, weight=None):
    """Cross-entropy over the K candidate logits -- nimble's objective, exactly.

    Nimble trains with plain CE on the candidate set: the model is asked to put
    mass on the correct code token, and that is the whole loss. No length
    normalisation, no margin, no auxiliary term. Keeping it identical is the
    point -- a different objective would make the comparison meaningless.

    The distribution comes out of the softmax at inference, which is what makes
    it a Janus-type read-out: probabilities over exactly the representable answers.
    """
    return F.cross_entropy(logits.float(), y, weight=weight)


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def load(name):
    p = os.path.join(ROOT, "data", "responses", f"{name}.jsonl")
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if r["y"] in CLASSES]


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("JANUS_BASE_MODEL",
                                                     "unsloth/Llama-3.2-1B"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out_janus"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load("train")
    held = load("rule_held")
    unm_path = os.path.join(ROOT, "data", "responses", "unmatched_held.jsonl")
    unm = [json.loads(l) for l in open(unm_path)] if os.path.exists(unm_path) else []

    print(f"classes {len(CLASSES)}: {CLASSES}")
    print(f"rows: train {len(tr)}  held {len(held)}  unmatched {len(unm)}")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"      # keep the failure text and answer cue

    base = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to(dev)
    model = get_peft_model(base, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    prompts = [build_prompt(r["x"]) for r in tr]
    ids = code_ids(tok, prompts[0], [OPTION_TEXT[c] for c in CLASSES])
    print(f"code ids at boundary: {ids}")
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=a.maxlen)
    y_tr = torch.tensor([CLASSES.index(r["y"]) for r in tr])
    cnt = collections.Counter(r["y"] for r in tr)
    cw = torch.tensor([math.sqrt(len(tr) / (len(CLASSES) * cnt[c]))
                       for c in CLASSES], dtype=torch.float32)
    print(f"class weights: {[f'{c}={w:.2f}' for c, w in zip(CLASSES, cw.tolist())]}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
    steps = int(math.ceil(len(tr) / a.bs * a.epochs))
    model.train()
    t0 = time.time()
    hist = collections.deque(maxlen=50)

    # SHUFFLE. Without it the rows are read in file order, and the data is
    # class-clustered: batch@400 was 6-of-8 `debug` while the rare classes sit
    # together around positions 170-230. Measured consequence was loss swinging
    # 0.71 -> 3.10 -> 3.03 between consecutive logging points, because one batch
    # is homogeneous and the next carries several weight-2.88 rare rows. That
    # trains on batch-order noise, not on the task.
    #
    # A seeded generator per epoch also means each seed sees a DIFFERENT order,
    # which the un-shuffled version did not: every seed consumed the same
    # sequence, so the three runs differed only in LoRA init and dropout. That
    # made n=3 measure init variance, which is the mistake run 29 already paid
    # for. Shuffling fixes both problems at once.
    order_rng = np.random.default_rng(a.seed)
    order = order_rng.permutation(len(tr))
    oi = 0
    epoch = 0
    for s in range(steps):
        if oi + a.bs > len(order):
            epoch += 1
            order = order_rng.permutation(len(tr))
            oi = 0
        idx = order[oi:oi + a.bs].tolist()
        oi += a.bs
        batch = {k: v[idx].to(dev) for k, v in enc.items()}
        out = model(**batch)
        last = batch["attention_mask"].sum(1) - 1
        logits = out.logits[torch.arange(len(idx)), last][:, ids]
        del out
        loss = nimble_loss(logits, y_tr[idx].to(dev), cw.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if (s + 1) % 100 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- evaluate on held-out sessions ------------------------------------
    @torch.no_grad()
    def predict(rows, bs=8):
        model.eval()
        out_p = []
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            ps = [build_prompt(r["x"]) for r in chunk]
            cd = code_ids(tok, ps[0], [OPTION_TEXT[c] for c in CLASSES])
            e = tok(ps, return_tensors="pt", padding=True, truncation=True,
                    max_length=a.maxlen).to(dev)
            o = model(**e)
            lastpos = e["attention_mask"].sum(1) - 1
            lg = o.logits[torch.arange(len(chunk)), lastpos][:, cd]
            del o
            out_p.append(torch.softmax(lg.float(), dim=-1).cpu().numpy())
        return np.concatenate(out_p, axis=0)

    PH = predict(held)
    yh = np.array([CLASSES.index(r["y"]) for r in held])
    pred = PH.argmax(1)
    acc = float((pred == yh).mean())
    maj = float((yh == collections.Counter(yh).most_common(1)[0][0]).mean())

    # proper scoring rules on the held set
    onehot = np.eye(len(CLASSES))[yh]
    brier = float(((PH - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(PH[np.arange(len(yh)), yh].clip(1e-12)).mean())

    # calibration
    conf = PH.max(1)
    correct = (pred == yh).astype(float)
    ece = 0.0
    for b in range(10):
        m = (conf >= b / 10) & (conf < (b + 1) / 10)
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())

    print(f"\n=== HELD-OUT (unseen sessions) n={len(held)} ===")
    print(f"  accuracy   {acc*100:5.1f}%   majority {maj*100:5.1f}%   "
          f"lift {(acc-maj)*100:+5.1f}pt")
    print(f"  brier      {parts['brier']:.4f}   nll {parts['nll']:.4f}   "
          f"ece {ece:.4f}")
    print(f"  mean conf  {conf.mean():.3f}")
    for k in sorted(set(yh)):
        m = yh == k
        print(f"    {CLASSES[k]:10s} n={m.sum():4d}  recall {100*(pred[m]==k).mean():5.1f}%")

    if unm:
        PU = predict(unm)
        cu = PU.max(1)
        print(f"\n=== UNMATCHED (rule could not label) n={len(unm)} ===")
        print(f"  mean conf {cu.mean():.3f}  confident(>0.6) {100*(cu>0.6).mean():.1f}%")
        print(f"  spread {dict(collections.Counter(np.array(CLASSES)[PU.argmax(1)]).most_common())}")

    res = {"seed": a.seed, "classes": list(CLASSES), "n_train": len(tr),
           "n_held": len(held), "held_acc": acc, "majority": maj,
           "brier": brier, "nll": nll, "ece": float(ece)}
    json.dump(res, open(f"{a.out}/result_seed{a.seed}.json", "w"), indent=2)
    model.save_pretrained(f"{a.out}/adapter_seed{a.seed}")
    print(f"\nsaved -> {a.out}/adapter_seed{a.seed}")


if __name__ == "__main__":
    main()
