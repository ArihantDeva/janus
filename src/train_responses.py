"""Train Llama-3.2-1B to classify a tool failure into its correct RESPONSE.

WHAT THIS IS
------------
The task the user actually asked for: a small classifier that reads a tool
failure and says what to do about it. Input is the error text. Output is one of

    retry      re-run UNCHANGED (the environment was slow)
    reauth     re-authenticate (credentials are wrong; retrying is futile)
    fix_call   the tool rejected the ARGUMENTS; fix the call
    fix_input  the path/handle does not exist; fix the input
    debug      the action is broken; read the traceback and change it
    abort      interrupted; stop

MECHANISM, REUSED DELIBERATELY
------------------------------
This is the letter-code read-out from run 31, which measured 89.3% accuracy and
cov@5 83% on an in-distribution task. The answer codes are the class names, and
the model scores each with a single-token letter at the answer boundary. No
generation, no parsing: the distribution is a softmax over K code logits.

The reason to reuse it rather than invent something: it is the only read-out in
this project that has ever produced a calibrated distribution, and it is the same
mechanism nimble uses. Changing objective and read-out at once would make the
result uninterpretable.

WHAT IS HELD OUT, AND WHY IT IS SPLIT THREE WAYS
-----------------------------------------------
Labels come from a rule over the same text the model reads, so a naive split
measures whether the model can reproduce the rule. Run 33 measured exactly that:
model accuracy 91.0% and model==rule agreement 91.0%, identical. This trainer
therefore reports three sets:

    train          rule-labelled rows
    rule_held      rule-labelled rows in unseen SESSIONS  (reproduces the rule?)
    unmatched_held rule could NOT label, never used to design rules (generalises?)

`unmatched` has no ground truth, so it is reported by confidence and class spread
rather than accuracy. Predicting those rows confidently would be the evidence of
generalisation; predicting them at chance confidence is the evidence against.

CLASS FILTER: classes with fewer than --min-class rows are dropped from training
and scoring. With 9 rows of `inspect`, a model cannot learn it and including it
only adds a class it will never predict correctly. Dropped classes are reported.
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

#: Fixed instruction. One place, so train and eval cannot disagree.
INSTRUCTION = (
    "You are diagnosing a failed tool call for a coding agent.\n"
    "Read the failure and decide what the agent should do next.\n"
    "Answer with the single letter of the best option."
)


def build_prompt(error_text, options):
    """Prompt + option block + answer cue. Ends on 'Answer: ' so the code is a
    standalone token -- the boundary requirement measured on Llama-3.2-1B."""
    lines = [f"{CODES[i]}) {o}" for i, o in enumerate(options)]
    return (f"{INSTRUCTION}\n\n"
            f"Failure:\n{error_text.strip()}\n\n"
            f"Options:\n" + "\n".join(lines) + "\n\nAnswer: ")


def option_text(cls):
    """The option string shown to the model, one per response class."""
    return {
        "retry": "the environment was slow - run the same call again unchanged",
        "reauth": "the credentials are wrong - re-authenticate before retrying",
        "fix_call": "the tool rejected the arguments - correct the call",
        "fix_input": "the path or handle does not exist - correct the input",
        "debug": "the action itself is broken - read the error and change the code",
        "abort": "the user interrupted - stop and do not retry",
    }[cls]


def load(name):
    p = os.path.join(ROOT, "data", "responses", f"{name}.jsonl")
    return [json.loads(l) for l in open(p)]


def keep_classes(rows, classes):
    return [r for r in rows if r["y"] in classes]


def encode(tok, rows, classes, maxlen):
    """Tokenise prompts and resolve the code ids at the real boundary."""
    prompts, ids_first = [], None
    opts = [option_text(c) for c in classes]
    for r in rows:
        p = build_prompt(r["x"], opts)
        prompts.append(p)
    # The code ids depend on the boundary, not the row, and the boundary is
    # identical for every row because the prompt ends with the same cue. Resolve
    # once and assert the first row agrees, rather than assuming.
    ids = code_ids(tok, prompts[0], opts)
    for p in prompts[:20]:
        if code_ids(tok, p, opts) != ids:
            raise SystemExit("code ids differ across rows; batching is unsafe")
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=maxlen)
    y = np.array([classes.index(r["y"]) for r in rows])
    return enc, ids, y


@torch.no_grad()
def predict(model, tok, rows, classes, maxlen, bs=8):
    model.eval()
    opts = [option_text(c) for c in classes]
    probs = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        prompts = [build_prompt(r["x"], opts) for r in chunk]
        ids = code_ids(tok, prompts[0], opts)
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        out = model(**enc)
        last = enc["attention_mask"].sum(1) - 1
        logits = out.logits[torch.arange(len(chunk)), last]
        del out
        picked = logits[:, ids].float()
        probs.append(torch.softmax(picked, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def report(tag, p, y, classes):
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    print(f"\n{tag}: n={len(y)}  acc {acc*100:5.1f}%")
    counts = collections.Counter(y)
    top = max(counts, key=counts.get)
    print(f"  majority({classes[top]}) {100*(y==top).mean():5.1f}%  "
          f"lift {(acc-(y==top).mean())*100:+5.1f}pt")
    for k in sorted(set(y)):
        m = y == k
        print(f"    {classes[k]:10s} n={m.sum():5d}  recall {100*(pred[m]==k).mean():5.1f}%")
    return acc


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
    ap.add_argument("--min-class", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(ROOT, "out_responses"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load("train")
    held = load("rule_held")
    unm = load("unmatched_held") if os.path.exists(
        os.path.join(ROOT, "data", "responses", "unmatched_held.jsonl")) \
        else load("unmatched")

    counts = collections.Counter(r["y"] for r in tr)
    classes = sorted(c for c, n in counts.items() if n >= a.min_class)
    dropped = sorted(c for c, n in counts.items() if n < a.min_class)
    print(f"kept classes : {classes}")
    print(f"dropped      : {dropped} (< {a.min_class} rows)")
    tr, held, unm = (keep_classes(x, classes) for x in (tr, held, unm))
    print(f"rows: train {len(tr)}  rule_held {len(held)}  "
          f"unmatched_held {len(unm)}")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if hasattr(tok, "truncation_side"):
        # keep the TAIL: the failure text and the answer cue are at the end
        tok.truncation_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to(dev)
    model = get_peft_model(base, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    enc_tr, ids, y_tr = encode(tok, tr, classes, a.maxlen)
    print(f"\ncode ids at the boundary: {ids}  ({len(classes)} classes)")

    lw = torch.tensor(
        [math.sqrt(len(tr) / (len(classes) * counts[c])) for c in classes],
        dtype=torch.float32, device=dev)
    print(f"class weights: {[f'{c}={w:.2f}' for c, w in zip(classes, lw.tolist())]}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
    steps = int(math.ceil(len(tr) / a.bs * a.epochs))
    model.train()
    t0, gi = time.time(), 0
    for s in range(steps):
        idx = [(gi + k) % len(tr) for k in range(a.bs)]
        gi += a.bs
        batch = {k: v[idx].to(dev) for k, v in enc_tr.items()}
        out = model(**batch)
        last = batch["attention_mask"].sum(1) - 1
        logits = out.logits[torch.arange(len(idx)), last]
        del out
        picked = logits[:, ids].float()
        tgt = torch.tensor(y_tr[idx], device=dev)
        loss = F.cross_entropy(picked, tgt, weight=lw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 100 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- evaluation -------------------------------------------------------
    ph = predict(model, tok, held, classes, a.maxlen)
    yh = np.array([classes.index(r["y"]) for r in held])
    acc = report("RULE_HELD (unseen sessions, rule-labelled)", ph, yh, classes)

    # coherence of the rule ON THE HELD SET, so "reproducing the rule" is visible
    from responses import classify_response
    rule_pred = [classify_response(r["x"])[0] for r in held]
    rule_ok = np.array([rp == r["y"] for rp, r in zip(rule_pred, held)])
    agree = float((np.array(classes)[ph.argmax(1)] == np.array(rule_pred)).mean())
    print(f"  rule itself scores {100*rule_ok.mean():.1f}% on these rows")
    print(f"  model agrees with rule on {100*agree:.1f}%  "
          f"(equal to accuracy => the model IS the rule)")

    pu = predict(model, tok, unm, classes, a.maxlen)
    conf = pu.max(1)
    print(f"\nUNMATCHED_HELD (n={len(unm)}): the rule could not label these")
    print(f"  mean confidence {conf.mean():.3f}   "
          f"confident(>0.6) {100*(conf>0.6).mean():.1f}%")
    print(f"  predicted spread: "
          f"{dict(collections.Counter(np.array(classes)[pu.argmax(1)]).most_common())}")

    res = {"seed": a.seed, "classes": classes, "dropped": dropped,
           "n_train": len(tr), "n_held": len(held), "n_unmatched": len(unm),
           "held_acc": acc, "rule_acc": float(rule_ok.mean()),
           "model_rule_agreement": agree,
           "unmatched_conf": float(conf.mean()),
           "unmatched_confident": float((conf > 0.6).mean())}
    with open(f"{a.out}/result_seed{a.seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    model.save_pretrained(f"{a.out}/adapter_seed{a.seed}")
    print(f"\nsaved -> {a.out}/adapter_seed{a.seed}")


if __name__ == "__main__":
    main()
