"""Train the Llama encoder classifier by distilling live Janus, then RLCD.

TWO STAGES, AND WHY BOTH ARE REAL
  stage 1  distillation: student KL-matches Janus's full distribution. This is what
           transfers the teacher's knowledge, including its calibration.
  stage 2  RLCD: for rows where Janus expressed a preference, the student is also
           pushed to separate the teacher's top choice from its runner-up. This is
           contrastive learning against the teacher's own near-misses, which is the
           RLCD shape: a positive and a plausible negative, not a random one.

Stage 2 is genuinely different from stage 1 -- `rlcd.contrastive_pairs` returns
None for near-uniform rows, so stage 2 acts on a strict subset and its effect is
measurable by comparing stage-1 and stage-2 checkpoints.

WHAT IS NOT DONE
No hard labels from the rule anywhere. The rule's label is carried in the corpus as
a comparison column so teacher-vs-rule disagreement stays measurable, but it never
enters the loss. Training on it would recreate run 33, where model==rule at 91.0%
and the model added nothing.

SPLIT
By `group` (project+session), so no session straddles train and eval. The assertion
is in code, not in a comment: a row-level split is the default mistake and it
inflates every number.

HONESTY CONTROLS BUILT IN
  tool-only floor     majority label per tool, computed on the eval rows
  keyword null        substring rules, same rows
  frozen-head control embedding frozen, only the 3-class head trained. If that
                      matches the LoRA runs, the LoRA was decoration.
All three are reported beside the model because a number with nothing next to it is
not a result.
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

import calibrate as CAL  # noqa: E402
import janus_interface as JI  # noqa: E402
import rlcd as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "distill", "janus.jsonl")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_corpus(path=CORPUS, min_conf=0.0):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "teacher" not in r:
            continue
        t = r["teacher"]
        if set(t) != set(JI.CLASSES_3):
            continue
        if max(t.values()) < min_conf:
            continue
        rows.append(r)
    return rows


def split_by_group(rows, frac=0.15, seed=0):
    """Group-disjoint train/eval. Asserts no group appears on both sides."""
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(groups))
    n_eval = max(1, int(round(frac * len(groups))))
    eval_g = {groups[i] for i in idx[:n_eval]}
    ev = [r for r in rows if r["group"] in eval_g]
    tr = [r for r in rows if r["group"] not in eval_g]
    overlap = {r["group"] for r in tr} & {r["group"] for r in ev}
    if overlap:                      # must never happen; asserted, not assumed
        raise AssertionError(f"group overlap between splits: {sorted(overlap)[:5]}")
    return tr, ev, sorted(eval_g)


def teacher_target(r):
    t = r["teacher"]
    v = np.array([t[c] for c in JI.CLASSES_3], dtype=np.float64)
    return v / v.sum()


# ---------------------------------------------------------------------------
# baselines, computed on the eval rows so the comparison is on identical data
# ---------------------------------------------------------------------------

def tool_only_floor(train, eval_rows):
    table = collections.defaultdict(collections.Counter)
    for r in train:
        table[r["tool"]][r["teacher_argmax"]] += 1
    best = {t: c.most_common(1)[0][0] for t, c in table.items()}
    default = collections.Counter(r["teacher_argmax"] for r in train).most_common(1)[0][0]
    return float(np.mean([best.get(r["tool"], default) == r["teacher_argmax"]
                          for r in eval_rows]))


def tfidf_ceiling(train, eval_rows, max_features=200000):
    """A linear model over the SAME rendered input. This is the bar that matters.

    Measured on the real corpus: word 1-2gram TF-IDF + logistic regression reaches
    **91.3%** held-out, against a 55.2% tool-only floor. So the task is far from
    impossible, and "beats the keyword null" is a trivial bar -- a regex null is not a
    competitor for a fine-tuned encoder. Any encoder claim must be stated against THIS
    number, or it is a claim that a 1.1B transformer outperformed `grep`.

    Returned as None when scikit-learn is unavailable so the trainer still runs on a
    cluster image without it; the number is then reported as "not measured" rather
    than silently omitted.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                          max_features=max_features)
    Xtr = vec.fit_transform([encoder_input(r, 4000) for r in train])
    Xev = vec.transform([encoder_input(r, 4000) for r in eval_rows])
    y_tr = [r["teacher_argmax"] for r in train]
    y_ev = [r["teacher_argmax"] for r in eval_rows]
    m = LogisticRegression(max_iter=2000, class_weight="balanced",
                           n_jobs=-1).fit(Xtr, y_tr)
    return float((m.predict(Xev) == np.array(y_ev)).mean())


def keyword_null(eval_rows):
    """The substring null, on the identical rows. Uniform => counts as abstention,
    and an abstention is scored as wrong here so the null cannot win by refusing."""
    import backends as B
    kb = B.KeywordBackend()
    hit = right = 0
    for r in eval_rows:
        d = kb(r["error"][:4000])
        top = max(d.values())
        if abs(top - min(d.values())) < 1e-9:
            continue                       # uniform: abstains
        pred = max(d, key=d.get)
        hit += 1
        right += (pred == r["teacher_argmax"])
    if not hit:
        return 0.0, 0.0
    return right / len(eval_rows), right / hit


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def tag_of(a):
    return f"freeze{a.seed}" if a.freeze_encoder else f"seed{a.seed}"


def save_checkpoint(model, tok, path):
    """Save encoder + tokenizer + head to one directory.

    NOT `model.save_pretrained`: the wrapper is a plain nn.Module, and calling that
    raised `AttributeError: 'Clf' object has no attribute 'save_pretrained'` AFTER
    stage 1 had finished, killing a 9-hour run at its first checkpoint. Save the
    encoder component explicitly, and the head separately because it is a bare
    nn.Linear that the encoder's own save does not cover.
    """
    import torch          # module-level torch is absent by design (tests run light)

    enc = getattr(model, "encoder", None)
    target = enc if hasattr(enc, "save_pretrained") else model
    target.save_pretrained(path)
    tok.save_pretrained(path)
    torch.save(model.head.state_dict(), os.path.join(path, "head.pt"))


def encoder_input(r, max_arg_chars=8000):
    st = JI.build_state(tool=r.get("tool"), arguments=r.get("arguments"),
                        error=r.get("error"))
    return JI.build_encoder_input(st, max_arg_chars=max_arg_chars)


def build_model(base_name, device, lora_r, freeze_encoder):
    import torch
    import torch.nn as nn

    import encoder_llama as E
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "right"

    # V100 (volta/cassini) has NO native bf16: it silently emulates, which is what
    # made the first cluster run take >6 min without completing 100 steps. Choose
    # the dtype from the actual device capability rather than assuming.
    # bf16 where native, else fp32. NOT fp16.
    #
    # Measured across two full restarts on V100 (sm70):
    #   1st attempt: bf16 on V100 (emulated)   -> 6 min, zero steps
    #   2nd attempt: fp16, no GradScaler       -> seed 2 NaN at step 5000
    #   3rd attempt: fp16 + GradScaler         -> seed 2 non-finite again at step 1771
    # So the scaler was necessary but not sufficient: a 0.34%-parameter LoRA over a
    # 16-layer bidirectional encoder produces gradient magnitudes that fp16's 5-bit
    # exponent cannot hold reliably, and the failure is seed-dependent, which means
    # ANY single-seed run could report a clean number over a corrupt model.
    #
    # bf16 has fp32's exponent range and is the only half precision worth using here.
    # On hardware that lacks it (V100, sm70) the honest choice is fp32: ~3x slower per
    # step, no silent NaN, and the run finishes inside the 12h wall either way. Speed
    # is not worth a result that might be a corrupted model.
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print(f"dtype {dtype} (device {device}, "
          f"capability {torch.cuda.get_device_capability() if device=='cuda' else 'n/a'})")
    enc = AutoModel.from_pretrained(base_name, dtype=dtype)
    # Gradient checkpointing is REQUIRED here, not an optimisation.
    #
    # Measured: bs=8 at maxlen 2048 with bidirectional attention allocated 30.28 GiB
    # and OOM'd on a 32 GB GPU, because without checkpointing every one of the 16
    # layers keeps its full activation stack alive for the backward pass, and a
    # non-causal attention pass has no causal shortcut to save memory. Checkpointing
    # recomputes activations in the backward pass, cutting the resident stack by
    # roughly the layer count at ~30% wall-clock cost -- the correct trade because
    # the alternative is not running.
    enc.gradient_checkpointing_enable()
    if hasattr(enc, "enable_input_require_grads"):
        enc.enable_input_require_grads()
    enc.config.use_cache = False
    before = E.count_causal_masks(enc)
    E.make_bidirectional(enc)
    E.assert_bidirectional(enc)
    print(f"causal masks: {before} before -> {E.count_causal_masks(enc)} after")

    class Clf(nn.Module):
        def __init__(self, encoder, hidden, n):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(hidden, n)

        def forward(self, input_ids, attention_mask):
            o = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = o.last_hidden_state.float()
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
            return self.head(pooled)

    model = Clf(enc, enc.config.hidden_size, len(JI.CLASSES_3)).to(device)

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("frozen encoder: only the 3-class head trains (the control)")
    elif lora_r > 0:
        from peft import LoraConfig, get_peft_model
        model.encoder = get_peft_model(model.encoder, LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        if hasattr(model.encoder, "enable_input_require_grads"):
            model.encoder.enable_input_require_grads()
    return model, tok, dtype


def metrics(P, target, ref_labels, floor=None):
    """Accuracy against the teacher argmax, plus calibration against the teacher.

    `ref_labels` are integer class indices. If they were strings the comparison
    silently produces 0% accuracy -- a defect that reads as a failed model rather
    than a failed report, so the type is asserted.
    """
    pred = P.argmax(1)
    ref = np.asarray(ref_labels)
    if ref.dtype.kind not in "iu":
        raise TypeError(
            f"reference labels must be integer class indices, got dtype {ref.dtype}. "
            f"String labels make `pred == ref` always False and report 0%.")
    acc = float((pred == ref).mean())
    conf = P.max(1)
    correct = (pred == ref).astype(float)
    ece = 0.0
    bins = []
    for b in range(10):
        m = (conf >= b / 10) & (conf < (b + 1) / 10)
        if m.sum():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap
            bins.append((b / 10, int(m.sum()), float(conf[m].mean()),
                         float(correct[m].mean())))
    tgt = np.asarray(target)
    kl = float((tgt * (np.log(np.clip(tgt, 1e-12, None))
                       - np.log(np.clip(P, 1e-12, None)))).sum(1).mean())
    return dict(acc=acc, ece=ece, kl=kl, bins=bins,
                per={c: (int((ref == i).sum()),
                         float((pred[ref == i] == i).mean()) if (ref == i).any() else None)
                     for i, c in enumerate(JI.CLASSES_3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("JANUS_BASE_MODEL",
                                                      "unsloth/Llama-3.2-1B"))
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--max-arg-chars", type=int, default=4000)
    ap.add_argument("--epochs1", type=float, default=3.0)
    ap.add_argument("--epochs2", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--rlcd-strength", type=float, default=0.5)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out_distill"))
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    # MPS matters: the cluster queues for hours on the constrained GPU nodes, and
    # the M1 Pro can run the identical script on the identical data. Same code path,
    # same numbers, so a local run is a real result and not a shortcut.
    # A GPU job that silently falls back to CPU is the worst outcome available: it
    # does not fail, it just runs ~100x slower and burns the wall clock. Measured: a
    # job on gpu017 printed `CUDA initialization: CUDA unknown error` to stderr and
    # then took 485s for ONE step. On a SLURM allocation, absence of CUDA is an
    # environment fault to report, never something to work around.
    if os.environ.get("SLURM_JOB_ID") and not torch.cuda.is_available():
        raise SystemExit(
            "SLURM allocation but torch.cuda.is_available() is False. Refusing to "
            "train on CPU: it would take ~100x longer and silently waste the "
            "allocation. Check the CUDA driver / CUDA_VISIBLE_DEVICES on this node.")
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    rows = load_corpus(a.corpus, a.min_conf)
    # THREE-WAY, by group. Temperature is fitted on `cal`, which is disjoint from both
    # train and eval: on train it would measure memorisation, on eval it would be
    # fitted-on-test and report an ECE the deployed model does not have.
    tr_all, ev, eval_groups = split_by_group(rows, seed=a.seed)
    cal_frac = 0.12
    tr, cal, _ = split_by_group(tr_all, frac=cal_frac, seed=a.seed + 500)
    CAL.assert_disjoint(tr, cal, ev)
    print(f"corpus {len(rows)} -> train {len(tr)} / calib {len(cal)} / eval {len(ev)} "
          f"({len(eval_groups)} held-out groups), disjoint asserted")

    # INDICES, not label strings. Measured bug: this held strings while `pred` was
    # an integer argmax, so `pred == ref` was False for every row and the first
    # stage-1 report read "acc 0.0%" with every per-class entry "n/a" -- a total
    # inversion presented as a catastrophic model. The float TE left below already
    # used indices; only the reference labels disagreed.
    y_ref = np.array([JI.CLASSES_3.index(r["teacher_argmax"]) for r in ev])
    T = np.stack([teacher_target(r) for r in ev])
    floor = tool_only_floor(tr, ev)
    kw_all, kw_ans = keyword_null(ev)
    print(f"teacher class balance (train): "
          f"{collections.Counter(r['teacher_argmax'] for r in tr).most_common()}")
    print(f"baselines on the SAME {len(ev)} eval rows:")
    print(f"  tool-only floor      {100*floor:5.1f}%   <- any claim must beat this")
    print(f"  keyword null         {100*kw_all:5.1f}% (answers {100*kw_ans:.0f}% of rows)")
    tfidf = tfidf_ceiling(tr, ev)
    if tfidf is None:
        print("  tf-idf linear        not measured (scikit-learn unavailable)")
    else:
        print(f"  tf-idf linear        {100*tfidf:5.1f}%  <- THE bar: a linear model on the "
              f"same input")

    # the interface constraint, before spending a GPU cycle
    for r in tr[:300]:
        JI.audit(encoder_input(r, a.max_arg_chars))
    print("interface audit: ok (state+question+choices, no cue, no instructions)")

    model, tok, dtype = build_model(a.model, dev, a.lora_r, a.freeze_encoder)

    def encode(rs):
        return tok([encoder_input(r, a.max_arg_chars) for r in rs],
                   return_tensors="pt", padding=True, truncation=True,
                   max_length=a.maxlen)

    # eval is encoded once (cheap, read-only); training rows are encoded per step.
    # Measured: materialising the whole corpus as one [N, maxlen] tensor OOM'd a
    # 32 GB GPU on seed 1 after seed 0 passed -- same script, same data, different
    # allocator state. Bounded per-step encoding removes the failure class.
    enc_ev = encode(ev)
    Ttr = np.stack([teacher_target(r) for r in tr])
    pairs, pair_idx = [], []
    for i, r in enumerate(tr):
        p = R.contrastive_pairs(teacher_target(r), None, JI.CLASSES_3)
        if p is not None:
            pairs.append(p)
            pair_idx.append(i)
    pairs = np.array(pairs) if pairs else np.zeros((0, 2), dtype=int)

    # GradScaler is REQUIRED on fp16 and must be absent on bf16.
    #
    # Measured: seed 2 (fp16 on a V100) went to `loss nan` at step 5000 while seeds
    # 0/1/3 stayed finite -- same code, same data, different seed. Under fp16 the
    # gradients of a small LoRA easily fall below the representable minimum and
    # underflow to zero, and once the head sees a zero gradient the weights drift
    # until the loss is NaN. bf16 has fp32's exponent range and needs no scaling;
    # using a scaler there is harmless but pointless, so it is gated on dtype.
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    print(f"grad scaler: {'on (fp16)' if scaler else 'off (bf16/fp32)'}")

    params = [p for p in model.parameters() if p.requires_grad]
    head_ids = {id(p) for p in model.head.parameters()}
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if id(p) not in head_ids], "lr": a.lr},
         {"params": [p for p in params if id(p) in head_ids], "lr": a.head_lr}],
        weight_decay=0.01)

    eval_chunks = [{k: v[i:i + 8].to(dev) for k, v in enc_ev.items()}
                   for i in range(0, enc_ev["input_ids"].shape[0], 8)]

    @torch.no_grad()
    def predict(_enc=None, bs=8):
        # eval rows are fixed, so their chunks are moved to the GPU ONCE. The
        # earlier version re-uploaded the whole eval set on every call, which
        # costs more than the forward pass it feeds.
        model.eval()
        out = [torch.softmax(model(**b).float(), -1).cpu().numpy()
               for b in eval_chunks]
        return np.concatenate(out, 0) if out else np.zeros((0, len(JI.CLASSES_3)))

    results = {}

    def flush(tag_name):
        """Write what is known so far.

        Measured risk this removes: v0 needs ~5.6h of a 6h wall clock, and results
        were only serialised after stage 2. A kill at 5h59m would have left NOTHING
        on disk -- no JSON, no checkpoint, six GPU-hours discarded. Persisting after
        each stage means the worst case is losing the stage in flight.
        """
        tag = f"freeze{a.seed}" if a.freeze_encoder else f"seed{a.seed}"
        partial = {"seed": a.seed, "n_train": len(tr), "n_eval": len(ev),
                   "tool_only_floor": floor, "keyword_null": kw_all,
                   "keyword_null_answered": kw_ans, "temp": a.temp,
                   "rlcd_strength": a.rlcd_strength, "classes": list(JI.CLASSES_3),
                   "freeze_encoder": a.freeze_encoder,
                   "n_contrastive_pairs": int(len(pairs)),
                   "results": results, "complete": False}
        json.dump(partial, open(f"{a.out}/result_{tag}.json", "w"), indent=2)
        return tag

    def run_stage(name, epochs, use_rlcd):
        steps = int(math.ceil(len(tr) / a.bs * epochs))
        rng = np.random.default_rng(a.seed + (1000 if use_rlcd else 0))
        order, oi, t0 = rng.permutation(len(tr)), 0, time.time()
        model.train()
        run = []
        nonfinite = [0]
        enc_cache = {}
        for s in range(steps):
            if oi + a.bs > len(order):
                order, oi = rng.permutation(len(tr)), 0
            idx = order[oi:oi + a.bs].tolist()
            oi += a.bs
            # encode at most once per distinct batch: epochs 2..N reuse the cache,
            # so tokenisation is paid once rather than at every step
            key = tuple(idx)
            if key in enc_cache:
                batch = enc_cache[key]
            else:
                batch = {k: v.to(dev) for k, v in
                         encode([tr[i] for i in idx]).items()}
                if len(enc_cache) < 3000:
                    enc_cache[key] = batch
            logits = model(**batch)
            soft = torch.tensor(R.soften(Ttr[idx], a.temp), dtype=torch.float32, device=dev)
            loss = F.kl_div(F.log_softmax(logits.float(), -1), soft,
                            reduction="batchmean")
            if use_rlcd and len(pairs):
                sel = [j for j, i in enumerate(pair_idx) if i in idx]
                if sel:
                    rows_i = [pair_idx[j] for j in sel]
                    pos = torch.tensor([pairs[j][0] for j in sel], device=dev)
                    neg = torch.tensor([pairs[j][1] for j in sel], device=dev)
                    posn = pos - min(pairs[:, 0].min(), 0)
                    # gather the teacher's top vs runner-up logits for these rows
                    li = torch.tensor([idx.index(i) for i in rows_i], device=dev)
                    lg = logits.float()[li]
                    pos_s = lg.gather(1, pos.view(-1, 1)).squeeze(1)
                    neg_s = lg.gather(1, neg.view(-1, 1)).squeeze(1)
                    loss = loss + a.rlcd_strength * torch.clamp(
                        0.5 - (pos_s - neg_s), min=0).mean()
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            if not torch.isfinite(loss):
                # A NaN step has already corrupted the weights. Loud, once, so the
                # run is not silently wasted; the per-stage checkpoint on disk is
                # still good, which is why it is written every stage.
                nonfinite[0] += 1
                if nonfinite[0] <= 3:
                    print(f"  !! non-finite loss at step {s+1} "
                          f"(count {nonfinite[0]}) -- weights may be corrupt",
                          flush=True)
            run.append(loss.item())
            if (s + 1) % 100 == 0 or s == 0:
                print(f"  [{name}] step {s+1}/{steps} loss {np.mean(run[-50:]):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        P = predict(enc_ev)
        m = metrics(P, T, y_ref, floor)
        m["nonfinite_steps"] = nonfinite[0]
        m["grad_scaler"] = scaler is not None
        results[name] = m
        print(f"  [{name}] eval acc {100*m['acc']:5.1f}% "
              f"(floor {100*floor:.1f}%, margin {100*(m['acc']-floor):+5.1f}pt)  "
              f"ece {m['ece']:.4f}  kl {m['kl']:.4f}")
        print(f"  [{name}] per-class: " + "  ".join(
            f"{k}={100*v[1]:.0f}%(n{v[0]})" if v[1] is not None else f"{k}=n/a"
            for k, v in m["per"].items()))
        # save the weights at each stage boundary too: stage 1 alone is a
        # reportable result (it is the distillation ablation), and a wall-clock
        # kill during stage 2 must not destroy it
        stage_dir = f"{a.out}/model_{'freeze' if a.freeze_encoder else 'seed'}{a.seed}_{name}"
        save_checkpoint(model, tok, stage_dir)
        flush(name)                      # durable immediately, not at the end
        return P

    P_final = None
    print(f"\n=== stage 1: distillation from Janus (T={a.temp}) ===")
    print(f"  contrastive pairs available for stage 2: {len(pairs)}/{len(tr)} "
          f"({100*len(pairs)/max(1,len(tr)):.0f}%)")
    run_stage("distill", a.epochs1, use_rlcd=False)
    print(f"\n=== stage 2: RLCD (strength {a.rlcd_strength}) ===")
    P_final = run_stage("rlcd", a.epochs2, use_rlcd=True)

    # ---- calibration (post-hoc temperature scaling, fitted on the calib split) ----
    with torch.no_grad():
        enc_cal = encode(cal)
        logits_cal = []
        for i in range(0, enc_cal["input_ids"].shape[0], 16):
            b = {k: v[i:i + 16].to(dev) for k, v in enc_cal.items()}
            logits_cal.append(model(**b).float().cpu().numpy())
        logits_cal = np.concatenate(logits_cal, 0)
        logits_ev = np.log(np.clip(P_final, 1e-12, None))   # recovered student logits
    y_cal = np.array([JI.CLASSES_3.index(r["teacher_argmax"]) for r in cal])
    T_fit = CAL.fit_temperature(logits_cal, y_cal, objective="ece")
    P_before = CAL.apply_temperature(logits_ev, 1.0)
    P_after = CAL.apply_temperature(logits_ev, T_fit)
    ece_before = CAL.expected_calibration_error(P_before, y_ref)
    ece_after = CAL.expected_calibration_error(P_after, y_ref)
    acc_after = float((P_after.argmax(1) == y_ref).mean())
    print(f"\n=== calibration (T fitted on {len(cal)} calib rows) ===")
    print(f"  T = {T_fit:.3f}")
    print(f"  ECE {ece_before:.4f} -> {ece_after:.4f}   "
          f"(target <= 0.05, {'PASS' if ece_after <= 0.05 else 'MISS'})")
    print(f"  accuracy unchanged by construction: {100*acc_after:.1f}% "
          f"(was {100*float((P_before.argmax(1) == y_ref).mean()):.1f}%)")
    results["calibration"] = {"T": T_fit, "ece_before": ece_before,
                              "ece_after": ece_after, "n_calib": len(cal),
                              "accuracy_after": acc_after}
    json.dump({"T": T_fit, "classes": list(JI.CLASSES_3)},
              open(f"{a.out}/temperature_{tag_of(a)}.json", "w"), indent=2)

    # out-of-distribution: tools never seen in training
    seen_tools = {r["tool"] for r in tr}
    ood = [i for i, r in enumerate(ev) if r["tool"] not in seen_tools]
    if ood:
        P = predict(enc_ev)
        ood_acc = float((P[ood].argmax(1) == np.array(y_ref)[ood]).mean())
        print(f"\nOOD: {len(ood)} eval rows on tools unseen in training -> "
              f"{100*ood_acc:.1f}%")
        results["ood"] = {"n": len(ood), "acc": ood_acc}

    tag = flush("final")
    res = json.load(open(f"{a.out}/result_{tag}.json"))
    res["complete"] = True
    json.dump(res, open(f"{a.out}/result_{tag}.json", "w"), indent=2)
    save_checkpoint(model, tok, f"{a.out}/model_{tag}")
    print(f"\nwrote {a.out}/result_{tag}.json")


if __name__ == "__main__":
    main()
