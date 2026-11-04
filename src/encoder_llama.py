"""Llama as a REAL encoder classifier: bidirectional, mean-pooled, headed.

WHY THIS IS NOT A FLAG ON AN LLM
A causal LM with a token read-out is still autoregressive machinery: it pays
prefill for the whole prompt so it can predict one next token. This module makes
Llama an encoder in the sense that matters:

  1. ATTENTION MASK -> NON-CAUSAL. Every layer's causal mask is replaced, so a
     token can attend to tokens after it. The error text sees the arguments and
     the choice criteria sees the error. For classifying a failure this is
     strictly more information than causal masking allows.
  2. NO NEXT-TOKEN POSITION. There is no answer cue and no code letter. The
     sequence is mean-pooled over the attention mask into one vector.
  3. THREE LINEAR HEADS. `hidden -> 3 logits`, softmax, done. The vocabulary is
     never touched; not one token is predicted, sampled or decoded.
  4. TRAINED BY CLASSIFICATION LOSS on the pooled vector, not by next-token
     cross-entropy on a letter.

This is the LLM2Vec construction (bidirectional conversion + pooling + heads)
applied to Llama-3.2-1B. The interface is unchanged: `janus_interface` still owns
the input, and the output is still {class: probability} -- the shape a Janus call
returns.

WHAT IS GIVEN UP, STATED PLAINLY
The token read-out can answer a question whose options were not known at training
time, because any letter is expressible. Fixed heads cannot: the option set is
baked into the head matrix. A Janus-style per-query option set therefore needs either
a head per question schema or a return to the token read-out. For this classifier
-- one question, three fixed choices -- the heads are the correct trade.

WHY THE MASK PATCH MUST BE VERIFIED
Silently leaving one causal mask in place produces a model that trains fine and
quietly ignores everything after each token. It would look like a working encoder
and behave like a worse causal LM. So `count_causal_masks` is reported before and
after, and `assert_bidirectional` refuses to train if any causal mask survives.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import janus_interface as JI  # noqa: E402

try:
    import torch
    import torch.nn as nn
except ImportError:  # cluster-only dependency; Mac-local suites skip torch tests
    torch = None
    nn = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# causal masking -> bidirectional
# ---------------------------------------------------------------------------

#: Attributes transformers has used to carry the masking direction, across
#: versions and attention implementations. Measured on the installed version:
#: LlamaAttention exposes `is_causal=True` and NO `causal` attribute, so a counter
#: that only looked for `causal` reported zero on a stock model -- a guard that
#: cannot fire, which is worse than no guard.
_CAUSAL_ATTRS = ("is_causal", "causal", "_is_causal", "causal_mask")


def _causal_modules(model):
    """Every module carrying a truthy causal flag, and which attribute it is."""
    out = []
    for name, mod in model.named_modules():
        for attr in _CAUSAL_ATTRS:
            try:
                val = getattr(mod, attr, None)
            except Exception:
                continue
            if val is True:
                out.append((attr, name, mod))
                break
    return out


def count_causal_masks(model):
    """How many modules still apply causal masking. Must reach 0 after patching."""
    return len(_causal_modules(model))


def make_bidirectional(model):
    """Turn every causal attention into full attention, in place.

    One mechanism only: clear the flag on each attention module. Verified
    behaviourally rather than assumed -- on the installed transformers, clearing
    `LlamaAttention.is_causal` is sufficient: a later token changes an earlier
    position's hidden state, which is impossible under causal masking (see
    `test_direction_actually_changes_the_output`).

    An earlier version also wrapped `forward` to force `is_causal=False` at call
    time. It was removed because it did not work here (attaching a closure to a
    `deprecate_kwarg`-decorated bound method bound `self` twice) and was not
    needed. If a future transformers re-enables masking inside the call, the
    behavioural test fails loudly rather than the model quietly staying causal --
    which is the whole reason that test exists.
    """
    patched = 0
    for attr, _name, mod in _causal_modules(model):
        try:
            setattr(mod, attr, False)
            patched += 1
        except Exception:
            pass
    return patched


def assert_bidirectional(model):
    left = count_causal_masks(model)
    if left:
        raise RuntimeError(
            f"{left} causal mask(s) survived patching. Training anyway would produce "
            f"a model that looks like an encoder and behaves like a worse causal LM.")


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

if nn is None:
    EncoderClassifier = None
else:
    class EncoderClassifier(nn.Module):
        """Bidirectional Llama + mean pooling + linear head. Single definition:
        training and inference share it so the architecture cannot disagree."""

        def __init__(self, encoder, hidden, n_cls):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(hidden, n_cls)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state.float()
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
            return self.head(pooled.to(self.head.weight.dtype))


def build_encoder(base_name, num_classes, device, lora_r=16):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = AutoModel.from_pretrained(base_name, torch_dtype=torch.bfloat16)
    n_causal = count_causal_masks(enc)
    patched = make_bidirectional(enc)
    assert_bidirectional(enc)
    print(f"causal masks: {n_causal} before -> {count_causal_masks(enc)} after "
          f"({patched} modules patched)")

    hidden = enc.config.hidden_size
    model = EncoderClassifier(enc, hidden, num_classes).to(device)

    if lora_r > 0:
        from peft import LoraConfig, get_peft_model
        model.encoder = get_peft_model(model.encoder, LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        if hasattr(model.encoder, "enable_input_require_grads"):
            model.encoder.enable_input_require_grads()
        print(f"LoRA r={lora_r} on the encoder; head is full-rank")
    return model, tok


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load(name, split_dir="responses_v3"):
    p = os.path.join(ROOT, "data", split_dir, f"{name}.jsonl")
    rows = [json.loads(l) for l in open(p)]
    # The deployed encoder head is a FROZEN 3-way softmax: rows carrying the
    # 4th Glimmer verdict (retry_wait) must never reach it. Filter on CLASSES_3
    # so the 3-class contract is enforced at the data boundary, not by trust.
    return [r for r in rows if r["y"] in JI.CLASSES_3]


def encoder_prompt(row, max_arg_chars=8000):
    """No answer cue: the pooled vector is the whole read-out."""
    st = JI.build_state(tool=row.get("tool"), arguments=row.get("arguments"),
                        error=row.get("error"))
    return JI.build_encoder_input(st, max_arg_chars=max_arg_chars)


def tool_only_floor(train, held):
    table = collections.defaultdict(collections.Counter)
    for r in train:
        table[r.get("tool")][r["y"]] += 1
    best = {t: c.most_common(1)[0][0] for t, c in table.items()}
    default = collections.Counter(r["y"] for r in train).most_common(1)[0][0]
    return sum(1 for r in held
               if best.get(r.get("tool"), default) == r["y"]) / max(1, len(held))


def metrics(P, y, classes):
    pred = P.argmax(1)
    acc = float((pred == y).mean())
    oh = np.eye(len(classes))[y]
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
        cov[str(tau)] = (float(m.mean()),
                         float(correct[m].mean()) if m.any() else float("nan"))
    per = {}
    for k, c in enumerate(classes):
        m = y == k
        if m.sum():
            per[c] = (int(m.sum()), float((pred[m] == k).mean()))
    return dict(acc=acc, brier=brier, nll=nll, ece=ece, cov=cov, per=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("JANUS_BASE_MODEL",
                                                      "unsloth/Llama-3.2-1B"))
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-arg-chars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out_encoder"))
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train, held = load("train"), load("rule_held")
    y_tr = np.array([JI.choice_index(r["y"]) for r in train])
    y_he = np.array([JI.choice_index(r["y"]) for r in held])
    floor = tool_only_floor(train, held)
    print(f"train {len(train)}  held {len(held)}  classes {JI.CLASSES_3}")
    print(f"tool-only floor on held: {100*floor:.1f}%  <- the margin that matters")

    for r in train[:200]:
        JI.audit(encoder_prompt(r, a.max_arg_chars))
    print("interface audit: ok (state+question+choices, no answer cue)")

    model, tok = build_encoder(a.model, len(JI.CLASSES_3), dev, a.lora_r)
    # right truncation drops the tail of a long argument blob, not the error
    tok.truncation_side = "right"

    def encode(rows):
        return tok([encoder_prompt(r, a.max_arg_chars) for r in rows],
                   return_tensors="pt", padding=True, truncation=True,
                   max_length=a.maxlen)

    # Chunked forward, NOT one [N, maxlen] tensor as input.
    #
    # Measured failure: seed 1 died with `Tried to allocate 4.00 GiB` after seed 0
    # had run fine, because the *input* encoding of the whole training set was
    # materialised as one tensor and pinned into the first CUDA allocation, and a
    # full encoder attention pass is O(batch * heads * seq^2) on top of it. Two
    # runs of the same script on the same data differing only by allocator state
    # is a memory bug, not bad luck.
    #
    # The pretokenised design also pinned every later step to the first stage's
    # window. Measuring at the end is cheap, so the eval set is encoded once and
    # the training set is encoded per step instead.
    def encode_rows(rows, maxlen):
        return tok([encoder_prompt(r, a.max_arg_chars) for r in rows],
                   return_tensors="pt", padding=True, truncation=True,
                   max_length=maxlen)
    params = [p for p in model.parameters() if p.requires_grad]
    head_ids = {id(p) for p in model.head.parameters()}
    groups = [
        {"params": [p for p in params if id(p) not in head_ids], "lr": a.lr},
        {"params": [p for p in params if id(p) in head_ids], "lr": a.head_lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    counts = collections.Counter(train[i]["y"] for i in range(len(train)))
    cw = torch.tensor([math.sqrt(len(train) / (len(JI.CLASSES_3) * counts[c]))
                       for c in JI.CLASSES_3], dtype=torch.float32, device=dev)
    print(f"class weights: {[f'{c}={w:.2f}' for c, w in zip(JI.CLASSES_3, cw.tolist())]}")

    steps = int(math.ceil(len(train) / a.bs * a.epochs))
    rng = np.random.default_rng(a.seed)
    order, oi, t0 = rng.permutation(len(train)), 0, time.time()
    model.train()
    for s in range(steps):
        if oi + a.bs > len(order):
            order, oi = rng.permutation(len(train)), 0
        idx = order[oi:oi + a.bs].tolist()
        oi += a.bs
        # encode just this step's rows: bounded memory, and the shortest sequence
        # in the batch sets the padding cost rather than the corpus maximum
        batch = {k: v.to(dev) for k, v in
                 encode_rows([train[i] for i in idx], a.maxlen).items()}
        logits = model(**batch)
        loss = F.cross_entropy(logits.float(),
                               torch.tensor(y_tr[idx], device=dev), weight=cw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 100 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def predict(rows, bs=8):
        model.eval()
        out = []
        for i in range(0, len(rows), bs):
            b = encode(rows[i:i + bs]).to(dev)
            out.append(torch.softmax(model(**b).float(), -1).cpu().numpy())
        return np.concatenate(out, 0)

    P = predict(held)
    m = metrics(P, y_he, JI.CLASSES_3)
    print(f"\n=== encoder, held-out ===")
    print(f"  acc {100*m['acc']:5.1f}%  (floor {100*floor:.1f}%, "
          f"margin {100*(m['acc']-floor):+5.1f}pt)")
    print(f"  ece {m['ece']:.4f}  brier {m['brier']:.4f}  nll {m['nll']:.4f}")
    print(f"  coverage@0.7 {100*m['cov']['0.7'][0]:5.1f}% at acc "
          f"{100*m['cov']['0.7'][1]:5.1f}%")
    print("  per-class: " + "  ".join(
        f"{k}={100*v[1]:.0f}%(n{v[0]})" for k, v in m["per"].items()))

    res = {"arch": "encoder-bidirectional-meanpool-head", "seed": a.seed,
           "n_train": len(train), "n_held": len(held), "tool_only_floor": floor,
           "maxlen": a.maxlen, "lr": a.lr, "classes": list(JI.CLASSES_3),
           "metrics": m}
    json.dump(res, open(f"{a.out}/result_seed{a.seed}.json", "w"), indent=2)

    import torch as _t
    _t.save({"head": model.head.state_dict(),
             "classes": list(JI.CLASSES_3)},
            f"{a.out}/head_seed{a.seed}.pt")
    (model.encoder if hasattr(model.encoder, "save_pretrained") else model).save_pretrained(
        f"{a.out}/encoder_seed{a.seed}")
    print(f"\nwrote {a.out}/result_seed{a.seed}.json, head + encoder")


if __name__ == "__main__":
    main()


def wrap_classifier(encoder, model_dir, hidden, n_classes):
    """Load a saved encoder + its tokenizer into the classifier head wrapper.

    Exists so inference and training cannot disagree about the architecture: the
    pooling and the head are the same two lines in both places, and both go through
    this function rather than being re-typed in the backend.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "right"
    return EncoderClassifier(encoder, hidden, n_classes), tok
