"""Train Llama-3.2-1B as a non-autoregressive tool-failure predictor.

Mechanism: no generation. We read the LM head's logits for the single-token
candidates " yes"/" no" at the final position and softmax over just those two.
p(fail) is that distribution. LoRA keeps optimizer state small enough to fit
one 44GB L40S (full-param AdamW OOMs at this size).

The {allow, ask, block} decision is NOT produced by the model - it is a
threshold on p(fail) applied in code, which is the whole point of a System One
split: the model supplies a probability, the caller decides.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Module scope: __file__ is only reliably bound at import time.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("JANUS_DATA", os.path.join(ROOT, "data"))


#: Which corpus to train on. The safety corpora and the SWE corpora are
#: different questions ("is this unsafe" vs "will this error") and must never be
#: mixed silently -- that mismatch is what made the first fine-tune score below
#: its floor on both safety benchmarks. Selecting it explicitly keeps the
#: objective visible at the call site.
CORPUS = os.environ.get("JANUS_CORPUS", "swe")


def load(split):
    return [json.loads(l) for l in open(f"{D}/{CORPUS}_{split}.jsonl")]


def cand_ids(tok):
    out = {}
    for w in ("yes", "no"):
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"'{w}' is {len(ids)} tokens - masking impossible")
        out[w] = ids[0]
    return out


@torch.no_grad()
def score_choice(model, tok, rows, bs=8, maxlen=512):
    """Per-row distribution over the K option codes, plus the derived p(unsafe).

    Reads ONLY the K candidate ids at the final position, exactly as the shipped
    head reads two. Rendering and id resolution both come from choice_task, so
    this cannot drift from the trainer.

    Rows are rendered one at a time and their ids resolved from that rendered
    prompt. Code ids are a property of the boundary, not of the option set, so
    resolving once globally would be the bug the parity test exists to catch.
    """
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        prompts, ids_list = [], []
        for r in chunk:
            prompt, _ = CT.build(r)
            prompts.append(prompt)
            ids_list.append(CT.resolve_ids(tok, prompt))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        res = model(**enc)
        last = (enc["attention_mask"].sum(1) - 1)
        row_logits = res.logits[torch.arange(res.logits.size(0)), last]
        del res
        for j, ids in enumerate(ids_list):
            # Comparing a LIST against one element is always unequal, so the
            # first version of this check fired on every batch (measured: it
            # killed the run at step 1). The real condition is whether any row
            # disagrees with the first.
            if any(x != ids_list[0] for x in ids_list):
                raise SystemExit(
                    "code ids differ between rows in one batch; batching is not "
                    "safe for this boundary and the read would be wrong")
            p = probs_from_logits(row_logits[j], ids).cpu().numpy()
            out.append(p)
    return np.asarray(out)


@torch.no_grad()
def score(model, tok, prompts, cids, bs=16, maxlen=256):
    model.eval()
    probs = []
    for i in range(0, len(prompts), bs):
        # Tokenizer is configured by keep_tail() to drop the head, not the tail:
        # the action and answer cue live at the end of the prompt.
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        # Only the last position matters, and only two vocabulary entries.
        # Taking the full [B,T,V] logits tensor costs ~13GB here and OOMs.
        out = model(**enc)
        last = (enc["attention_mask"].sum(1) - 1)
        pair = out.logits[torch.arange(out.logits.size(0)), last][
            :, candidates(cids)].float()
        del out
        probs.extend(p_fail_from_pair(pair.cpu().numpy(), cids).tolist())
    return np.array(probs)


# Metrics come from the shared module. A local copy is how the README ends up
# disagreeing with the tests: two implementations, two answers. The trainer
# only trains and scores; every published number is computed by metrics.py.
from noul import candidates, p_fail_from_pair, target_index  # noqa: E402
import choice_task as CT  # noqa: E402
from choice_codes import probs_from_logits  # noqa: E402
from train_util import ensure_pad, keep_tail
from metrics import (
    kl_from_prior, memorisation_gap,          # noqa: E402
    apply_temperature as apply_T,
    fit_temperature as fit_T,
    gate as gate_metrics,
    summarize,
)

def report(tag, p, y):
    s = summarize(p, y)
    print(f"  {tag:10s} acc {s['acc']*100:5.1f}%  brier {s['brier']:.4f}  "
          f"ece {s['ece']:.4f}  aurc {s['aurc']:.4f}  "
          f"cov@5 {s['cov5']*100:4.0f}%  cov@10 {s['cov10']*100:3.0f}%  "
          f"cov@20 {s['cov20']*100:3.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    # Class weighting, never tested in 18 runs. The corpora are 55.0% unsafe
    # while the benchmarks are 64.9% (rjudge) and 63.3% (asb) unsafe, so an
    # unweighted objective puts the decision boundary ~9-10pt too far toward
    # "unsafe" for the evaluation population. Weighting changes GRADIENTS, so
    # unlike a post-hoc threshold it can move AUC, not just the operating point.
    #
    # "sqrt" = weights proportional to 1/sqrt(class frequency). Chosen a priori
    # over full inverse-frequency because full balancing severely distorts an
    # already-small training signal and was the kind of over-correction that
    # collapsed the model in run 6.
    # Which answer encoding. "noul" is the shipped 2-token " yes"/" no" softmax
    # and stays the default so every run before this one remains reproducible
    # byte for byte. "choice" is the same mechanism widened to K letter codes,
    # which is the only way a third outcome (ask) becomes representable.
    # The option set, label mapping and p(unsafe) rule live in choice_task.py so
    # training and evaluation cannot disagree about them.
    ap.add_argument("--task", default="noul", choices=("noul", "choice"))
    ap.add_argument("--class-weight", default="none",
                    choices=("none", "sqrt", "inv"))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=None,
                    help="comma list, e.g. q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--out",
                    default=os.environ.get("JANUS_OUT", os.path.join(ROOT, "out")))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.model)
    ensure_pad(tok)
    keep_tail(tok)
    cids = cand_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    # LoRA width and target set are flags, not constants, so an experiment can
    # change them without editing this file. A loop that edits source between
    # runs cannot tell a hyperparameter effect from a code change.
    tm = [t for t in a.target_modules.split(",") if t] if a.target_modules else \
        ["q_proj", "k_proj", "v_proj", "o_proj"]
    model = get_peft_model(base, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=tm))
    model.gradient_checkpointing_enable()
    # LoRA freezes the base embeddings, so checkpointing would otherwise get a
    # graph with no grad_fn. Force the input embeddings to require grad.
    model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    tr, dv, te = load("train"), load("dev"), load("test")
    print(f"seed={a.seed} device={dev} train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)

    zy = np.array([r["label"] for r in te])
    xt = [r["prompt"] for r in te]
    zp = (CT.p_unsafe_from_probs(score_choice(model, tok, te, maxlen=a.maxlen))
          if a.task == "choice"
          else score(model, tok, xt, cids, maxlen=a.maxlen))

    # ---- class weights (see --class-weight) --------------------------------
    cw = None
    if a.class_weight != "none":
        lbl = np.array([int(r["label"]) for r in tr])
        freq = np.array([(lbl == 0).sum(), (lbl == 1).sum()], float) / len(lbl)
        if a.class_weight == "sqrt":
            w = 1.0 / np.sqrt(np.maximum(freq, 1e-9))
        else:
            w = 1.0 / np.maximum(freq, 1e-9)
        w = w / w.mean()                    # keep overall loss scale comparable
        cw = torch.tensor(w, dtype=torch.float32)
        print(f"class_weight={a.class_weight} unsafe_freq={freq[1]:.3f} "
              f"w_safe={w[0]:.3f} w_unsafe={w[1]:.3f}", flush=True)

    # Only rows that actually carry a label can supervise the choice head.
    tr_labeled = [r for r in tr if r.get("label") is not None]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = int(math.ceil(len(tr) / a.bs * a.epochs))
    model.train()
    t0, gi = time.time(), 0
    for s in range(steps):
        b = [tr[(gi + k) % len(tr)] for k in range(a.bs)]
        gi += a.bs
        # Same side as scoring: keep the tail, where the action and answer cue
        # live. Training and scoring MUST agree here or the model learns from
        # one region of the prompt and is judged on another.
        if a.task == "choice":
            prompts_b = []
            ids_list_b = []
            for r in b:
                pr, _ = CT.build(r)
                prompts_b.append(pr)
                ids_list_b.append(CT.resolve_ids(tok, pr))
        else:
            prompts_b = [r["prompt"] for r in b]
        enc = tok(prompts_b, return_tensors="pt", padding=True,
                  truncation=True, max_length=a.maxlen).to(dev)
        out = model(**enc)
        last = (enc["attention_mask"].sum(1) - 1)
        row_logits = out.logits[torch.arange(len(b)), last]
        del out
        if a.task == "choice":
            # Cross-entropy over the K candidate code logits. Same form as the
            # noul path, K instead of 2 -- nimble's objective.
            if any(x != ids_list_b[0] for x in ids_list_b):
                raise SystemExit(
                    "code ids differ within a batch; the read would be wrong")
            picked = row_logits[:, ids_list_b[0]].float()
            tgt = torch.tensor([CT.option_for_label(r["label"]) for r in b],
                               device=dev)
            # ASK has no supervision by construction, so the class weights index
            # the K option classes rather than the 2 labels.
            cwk = None
            if a.class_weight != "none":
                opt_freq = np.zeros(len(CT.OPTIONS))
                for r in tr_labeled:
                    opt_freq[CT.option_for_label(r["label"])] += 1
                opt_freq = opt_freq / opt_freq.sum()
                w = 1.0 / (np.sqrt(np.maximum(opt_freq, 1e-9))
                           if a.class_weight == "sqrt"
                           else np.maximum(opt_freq, 1e-9))
                # An unsupervised option must not be up-weighted: with zero
                # rows its weight is infinite and would dominate the loss on
                # whichever rows came closest. Pin it to 0 instead.
                w = np.where(opt_freq > 0, w, 0.0)
                w = w * (opt_freq > 0)
                w = w / w[opt_freq > 0].mean() if (opt_freq > 0).any() else w
                cwk = torch.tensor(w, dtype=torch.float32, device=picked.device)
                # A zero weight on the row's OWN class makes PyTorch return nan
                # (measured), which poisons every parameter in one step. With
                # only supervised options ever used as targets this cannot fire,
                # but it is one relabelling away from a silent total loss.
                if (cwk[tgt] == 0).any():
                    raise SystemExit(
                        "a training row targets an option with zero class "
                        "weight; cross_entropy would return nan")
            loss = F.cross_entropy(picked, tgt, weight=cwk)
        else:
            pair = row_logits[:, candidates(cids)].float()
            tgt = torch.tensor([target_index(r["label"], cids) for r in b],
                               device=dev)
            if cw is None:
                loss = F.cross_entropy(pair.float(), tgt)
            else:
                loss = F.cross_entropy(pair.float(), tgt,
                                       weight=cw.to(pair.device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 200 == 0 or s == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    if a.task == "choice":
        print("\n=== CHOICE TASK ===")
        print(CT.describe(), flush=True)
        dvp = CT.p_unsafe_from_probs(score_choice(model, tok, dv, maxlen=a.maxlen))
        T = fit_T(dvp, np.array([r["label"] for r in dv]))
        tep = score_choice(model, tok, te, maxlen=a.maxlen)
        tp = CT.p_unsafe_from_probs(tep)
        tpc = apply_T(tp, T)
        dist = tep.mean(axis=0)
        print(f"  mean distribution ALLOW/ASK/BLOCK "
              f"{[f'{v:.3f}' for v in dist]}")
        chosen = np.bincount(tep.argmax(axis=1), minlength=len(CT.OPTIONS)) / len(tep)
        print(f"  chosen share   {[f'{v:.3f}' for v in chosen]}")
        if chosen[CT.I_ASK] == 0:
            print("  note: ASK never chosen. Expected -- it has no training rows.")
    else:
        dp = score(model, tok, [r["prompt"] for r in dv], cids, maxlen=a.maxlen)
        T = fit_T(dp, np.array([r["label"] for r in dv]))
        tp = score(model, tok, xt, cids, maxlen=a.maxlen)
        tpc = apply_T(tp, T)

    # Score a TRAIN sample so the memorisation gap is measurable. A training
    # loss falling to 0.01 proves nothing on its own: a model that memorises
    # reaches it while generalising no better than chance. The gap between
    # in-sample and held-out information is the signal.
    n_gap = min(1500, len(tr))
    gap_rows = tr[:n_gap]
    gp = (CT.p_unsafe_from_probs(score_choice(model, tok, gap_rows,
                                             maxlen=a.maxlen))
          if a.task == "choice"
          else score(model, tok, [r["prompt"] for r in gap_rows], cids,
                     maxlen=a.maxlen))
    gy = np.array([r["label"] for r in gap_rows])
    mem = memorisation_gap(gp, gy, tp, zy)
    print(f"\n=== MEMORISATION (train n={n_gap} vs test n={len(te)}) ===")
    print(f"  KL over prior  train {mem['kl_train']:+.4f}  "
          f"test {mem['kl_test']:+.4f}  gap {mem['kl_gap']:+.4f}")
    print(f"  accuracy       train {mem['acc_train']*100:5.1f}%  "
          f"test {mem['acc_test']*100:5.1f}%  gap {mem['acc_gap']*100:+5.1f}pt")
    if mem["flagged"]:
        print("  WARNING: train is markedly better than held-out. This is the "
              "memorisation signature, not learning.")
    else:
        print("  gap is small: the model is not merely reproducing seen rows")

    print(f"\n=== TEST n={len(te)} fail={int(zy.sum())} T={T:.2f} ===")
    report("zero-shot", zp, zy)
    report("lora", tp, zy)
    report("lora+T", tpc, zy)
    print("\n=== GATE SWEEP on lora+T (block=hi, ask=lo..hi, allow<lo) ===")
    print(f"  {'hi':>4} {'lo':>4} {'block%':>7} {'ask%':>6} {'recall':>7} {'prec':>6} {'falseblk':>9}")
    for hi, lo in ((0.9, 0.7), (0.8, 0.5), (0.7, 0.3), (0.6, 0.2), (0.5, 0.2)):
        g = gate_metrics(tpc, zy, hi, lo)
        print(f"  {hi:4.1f} {lo:4.1f} {g['block_rate']*100:6.1f}% {g['ask_rate']*100:5.1f}% "
              f"{g['fail_recall']*100:6.1f}% {g['block_precision']*100:5.1f}% "
              f"{g['false_block_rate']*100:8.1f}%")
    g = gate_metrics(tpc, zy, 0.7, 0.3)

    res = {"seed": a.seed, "task": a.task, "T": T, "n": len(te),
           "options": list(CT.OPTIONS) if a.task == "choice" else None,
           "zero_shot": {"acc": float(((zp >= .5).astype(int) == zy).mean()),
                         "brier": float(np.mean((zp - zy) ** 2)), "ece": summarize(zp, zy)["ece"]},
           "lora": {"acc": float(((tp >= .5).astype(int) == zy).mean()),
                    "brier": float(np.mean((tp - zy) ** 2)),
                    **{k: v for k, v in summarize(tp, zy).items()
                       if k in ("ece", "aurc", "cov5")}},
           "lora_T": {"acc": float(((tpc >= .5).astype(int) == zy).mean()),
                      "brier": float(np.mean((tpc - zy) ** 2)),
                      **{k: v for k, v in summarize(tpc, zy).items()
                         if k in ("ece", "aurc", "cov5", "cov10", "cov20")}},
           "gate": g, "memorisation": mem}
    with open(f"{a.out}/lora_seed{a.seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    model.save_pretrained(f"{a.out}/adapter_seed{a.seed}")
    print(f"\nsaved -> {a.out}/adapter_seed{a.seed}", flush=True)


if __name__ == "__main__":
    main()
