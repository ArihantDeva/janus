# Autoresearch: Janus-distilled Llama classifier — beyond the frozen-control 82.20%

## Objective

Fine-tune a **frozen-encoder + 3-class head** (Llama-3.2-1B backbone, bidirectional, mean-pool)
to classify failed tool calls as **debug / fix_input / fix_call**, by distilling live Janus's
full probability distributions, then a contrastive RLCD stage. Beat the frozen-control
baseline on group-held-out Janus agreement **without memorizing**, and lift the weakest
class recall.

## Metrics

- **Primary**: `rlcd_acc` (fraction 0–1, higher is better) — Janus-argmax agreement after
  the RLCD stage, on the group-held-out eval split.
- **Secondary** (every run reports all of these):
  `distill_acc`, `rlcd_ece`, `distill_ece`, `kl`,
  `per_debug`, `per_fix_input`, `per_fix_call` (recall per class),
  `floor` (tool-only majority baseline, same rows), `margin` (rlcd_acc − floor),
  `ood_acc` + `ood_n`, `n_train`, `n_eval`, `nonfinite_steps`.

## Baselines (measured, frozen in history — do not re-tune to beat them by cheating)

| Run | distill_acc | rlcd_acc | rlcd_ece | debug | fix_input | fix_call | floor | ood |
|---|---|---|---|---|---|---|---|---|
| freeze0 (HEAD-ONLY CONTROL) | 0.8534 | **0.8220** | 0.1027 (0.0316 cal) | 0.606 | 0.915 | 0.888 | 0.5519 | 1.00 (n=11) |
| seed0 (LoRA) | 0.7674 | 0.6701 | 0.1109 | 0.707 | 0.812 | 0.595 | 0.5519 | 0.82 (n=11) |
| seed1 (LoRA) | 0.5113 | 0.3894 | 0.2247 | 0.980 | 0.000 | 0.354 | 0.4599 | 0.25 (n=4) |
| seed2 (LoRA) | 0.6154 | 0.5257 | 0.1062 | 0.619 | 0.759 | 0.326 | 0.4814 | 0.44 (n=9) |

Key facts: stage-2 RLCD **hurts** agreement in every run (freeze0: −3.1pt, seeds worse).
TF-IDF linear on the same input reaches **91.3%** — the task is learnable, the
representation is fine, the training recipe is the suspect. Seed-0 stage-1 artifact
provenance is unresolved after output collisions; treat seed0 numbers as suspect.

## How to Run

```
./.auto/measure.sh          # submits one SLURM job (train + eval), waits, emits METRIC lines
./.auto/checks.sh           # correctness gate, runs automatically after a passing measure
```

One iteration is one full train (2+1 epochs) ≈ 8–12h wall on Amarel (fp32 path on V100,
bf16 on sm80+). Slow workload — single runs, no median-of-N. **Batch mode**: the
queue fits ~8 concurrent 1-GPU jobs, so fire the decisive ablations as a parallel
batch and read them as a group in the morning. 2026-10-29 batch (all FREEZE=1,
seed 0, 2+1 epochs unless noted): baseline (strength 0.5), distill-only
(strength 0/epochs2 0), strength 0.1, strength 0.2, minconf06 (distill-only +
MIN_CONF=0.6), temp10 (T=1.0), headlr (HEAD_LR=3e-3). Jobs 63214152, 63214207,
63214214/16/17/20/22/23; all PENDING at batch time, muse-encoder job owns the
running GPU. Runs after this batch: winner re-seed (idea needs ±14pt confirmation),
then class-balanced/dropout and LoRA-low-LR only if head-only saturates.

Hyperparameters are **environment variables**, never source edits:

```
SEED FREEZE EPOCHS1 EPOCHS2 RLCD_STRENGTH TEMP MIN_CONF HEAD_LR LORA_R
MAXLEN MAX_ARG_CHARS OUT_SUFFIX EXTRA_ARGS
```

The loop must be able to tell a hyperparameter effect from a code change. Source edits
to `src/train_distill.py` are allowed only for principled algorithmic improvements
(head architecture, loss, sampling); they must be committed as the experiment.

## Files in Scope

| Path | Role |
|---|---|
| `src/train_distill.py` | The trainer: distillation (KL vs Janus soft targets) + RLCD contrastive stage, temperature calibration on a third split, OOD report |
| `src/rlcd.py` | Contrastive pair construction + margin/hinge losses (pure functions, unit-tested) |
| `src/calibrate.py` | Temperature fitting (ECE objective) — fitted on calib split only |
| `src/janus_interface.py` | State+question+choices input contract — audit owns this, do not relax it |
| `scripts/train_distill.sbatch` | Reference SLURM wrapper for standalone runs (loop uses its own sbatch) |

## Off Limits

- `data/responses_v3/*.jsonl` and anything derived from them as new training rows.
  The eval split is defined by `split_by_group` with the run's seed — do not
  hand-pick eval rows, do not tune on eval content, do not report a number without
  the floor beside it.
- **Eval rows are read-only.** Never early-stop on eval agreement, never pick
  checkpoints by eval agreement, never sweep hyperparameters against eval and keep
  the best. One eval read per run, reported, not optimized. Checkpoint and
  hyperparameter decisions use the calib split or rules fixed BEFORE submission.
- **One seed per idea, then confirm.** A single-seed win is a candidate, not proof:
  seed variance here is ±14pt. Confirm a kept idea on a second seed before calling
  it real. A win that vanishes on re-seed is noise — discard it.
- **Distill-only is the ablation, not a loophole.** `EPOCHS2=0` runs are legitimate
  ablations of the stage-2 hypothesis, reported beside the floor. They do not get
  extra epochs, extra data, or re-tuned temperatures to inflate them past RLCD runs.
- `src/janus_interface.py` input contract — state+question+choices, no system prompt,
  no instructions, no CoT. If you change the input, you changed the product.
- The majority-class floors, the ECE bound, and the group-disjointness assertion.
  Changing the bar to pass the bar is not an experiment.
- `/scratch/ad2416/out_distill/*` and `/scratch/ad2416/out_fast/*` — frozen history.
  All loop outputs go to `/scratch/ad2416/out_loop_*`.
- `/scratch/ad2416/janussrc` — the active training tree. The loop syncs this repo's
  `src/` to `/scratch/ad2416/janussrc-loop/` and trains there.
- Never `git push`, never cancel jobs you did not submit (63179577 family is history).

## Constraints

- `mise exec -- make check` must stay green (or document the pre-existing failure).
- No new runtime dependencies. The stack is `torch`, `transformers`, `peft`,
  `scikit-learn` (eval-only ceiling).
- All model compute on Amarel through SLURM. The Mac is for editing and SSH only —
  never load Torch or the model locally.
- A SLURM allocation without CUDA is an environment fault: refuse to train on CPU.
- Batch size and maxlen must respect the measured 30.28 GiB ceiling (bs=8 @ maxlen
  2048 with gradient checkpointing); bs=8 @ maxlen 8192 needs verification.
- Preserve splits, checkpoints, logs, provenance: every run's out dir, sbatch script,
  and result JSON are the evidence. Never overwrite prior loop outputs.

## What's Been Tried

### Baselines that set the problem up (pre-loop)

- TF-IDF linear 91.3% vs tool-only floor ~55% → the input contains the signal; the
  neural recipe is under-delivering, not the data.
- Frozen-head control (85.3% distill → 82.2% RLCD) beats every LoRA run → the
  encoder already separates the classes; LoRA updates are hurting, not helping.
- RLCD stage hurts agreement in ALL runs (−3 to −30pt) → the contrastive margin
  (teacher top-vs-runner-up hinge, strength 0.5) is the prime suspect. Either the
  strength is too high, the margin too aggressive, or the runner-up pairs are noise.
- Seed1 fix_input recall = 0.000 → collapse signature; seed0/seed2 trade debug
  against fix_call. The head is starved on minority-class gradients.
- Fast loop (out_fast) reproduced the same pattern at lower accuracy → not a fluke.

### Ideas backlog (highest value first)

1. **RLCD strength → 0 / epochs2 → 0 (distill-only ablation).** If distill-only holds
   ~85%, the entire stage-2 is the regression. Cheapest decisive experiment.
2. **RLCD strength sweep (0.1, 0.2).** Keep the contrastive signal but stop it from
   dominating the KL. Tests whether the recipe is salvageable vs. dead.
3. **MIN_CONF filtering (0.6).** Train only on rows where Janus expressed a real
   preference; near-uniform rows contribute gradient noise to both stages.
4. **Head LR up (2e-3–5e-3), encoder LR down or frozen.** The head is a fresh random
   init; the encoder is pretrained. Same LR for both is unjustified.
5. **Class-balanced sampling / head dropout.** fix_input collapses under imbalance
   (bash = 88% of rows). Balanced batches or 0.1–0.3 head dropout are cheap.
6. **TEMP 1.0 ablation.** T=2.0 flattens an already-confident teacher (median 0.97);
   may be washing out the runner-up distinctions RLCD needs.
7. Only after head-only saturates: **LoRA r sweep with low encoder LR** — the
   frozen result says the encoder needs little movement; large-LR LoRA destroys it.
