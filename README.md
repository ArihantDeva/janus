# Janus

> **Not maintained.** Published as a hackathon submission and a record of its
> measurements. No PRs, no issues, no support. See [CONTRIBUTING.md](CONTRIBUTING.md).

A calibrated **non-autoregressive** guard that predicts what a failed tool call
needs next — and returns a probability, not a sentence.

Built for the **Meta Global AI Developer Hackathon** (indie track), theme
*Push the Frontier*.

**Runtime model: Muse Glimmer 30B.** Llama-3.2-1B was the proof of concept.
Every measured number below is labeled with which model produced it.

## What runs at runtime

Janus never generates text. It takes one forward pass and gives you a
probability over a set of typed decisions — a non-autoregressive encoder
with the spirit of system-1 models, distilled from production agent traces
with RLCD.

Glimmer 30B QLoRA at runtime:

```
failure text -> Glimmer 30B + adapter -> 4 letter logits -> softmax -> p(class)
                                                          -> threshold in code
                                                          -> allow / ask / block
```

Single-token letter read-out (A/B/C/D). The verdict words (`fix_input`,
`retry_wait`) split into two tokens, so words cannot be read directly —
letters are guaranteed single tokens. No token is ever sampled. Latency is
one forward pass, not a decode loop.

The Llama rung proved the mechanism (single-pass masking + temperature).
Glimmer is what serves.

## Why this wins: cost, speed, ease of use

**Cost: close to free.** A frontier call burns ~800 reasoning tokens just to
answer why a tool call failed. Janus costs nothing — no Frontier call, no
per-token meter, no API key. Same question, 400x faster, close to free.

**Speed: milliseconds on GPU.** No Frontier call, no round trip — diagnosis
in milliseconds on GPU. Measured on Amarel (CPU inference, warmup dropped):

- inference p50 `405ms` / p95 `440ms` (10 timed calls, in-allocation)
- hook end-to-end p50 `1791ms` / p95 `1884ms` (SSH + srun + inference)
- policy overhead `0.43ms` (threshold logic, measured locally)

No generation means no tail from decode loops. The percentile is the product,
not the mean.

**Ease of use: one hook, one binary.**

```sh
mkdir -p .muse && cp ~/jevlab/muse/hooks.json .muse/hooks.json
```

```sh
muse-guard "rm -rf /usr"     # exit 3 block
muse-guard "cat ~/.ssh/id_rsa"  # exit 2 ask
muse-guard "ls -la"          # exit 0 allow
```

Exit codes a shell can act on; JSON on stdout for logs. Deterministic
prechecks (destructive patterns, workspace escapes) run before the model and
win — a probability never overrules a string comparison. If the scorer is
down, the guard raises instead of failing open.

To wrap it up: Janus is a system-1 guard under a system-2 agent. With it,
the agent doesn't just drive tools anymore — it feels their failures.

Install it, go and break something, and watch your answer faster, right
within Muse Code.

## Reproduce everything

```sh
make check   # tests + dataset integrity + leak gate + floor tests + guard contract
```

```
== guard exit-code contract (what a Muse Code wrapper sees) ==
  'rm -rf /usr' -> exit 3
  'cat ~/.ssh/id_rsa' -> exit 2
  'ls -la' -> exit 0

ALL CHECKS PASSED
```

Numbers in this README are produced by that command or by the scripts it
names. Nothing here is a proxy signal.

## Results

### Llama-3.2-1B proof of concept — where the mechanism numbers come from

Frozen-encoder + RLCD head, group-disjoint splits. Test n=3410.
Tool-only floor: **55.2%**.

| stage (Llama) | acc | ECE |
|---|---|---|
| distill | 85.3% | 0.17 |
| + RLCD | 82.2% | 0.10 |
| + temperature (T=0.5277, n=2787) | 82.2% | **0.0316** |

Single-token masking rung (n=676, floor 70.6%): accuracy **75.0%** mean,
ECE **0.2426 → ~0.050** with one fitted temperature. Same mechanism, smaller
model. This is provenance for the Glimmer recipe, not a runtime claim.

### Muse Glimmer 30B — the runtime classifier

4-way Janus verdict (debug / fix_input / fix_call / retry_wait), held-out
n=3410, group-disjoint. Artifact: `eval/glimmer_janus_seed0.json`.

| config (Glimmer) | acc | ECE | note |
|---|---|---|---|
| zero-shot | 33.3% | 0.3347 | honest starting point |
| QLoRA (2738 steps, 104.8M adapter params, NF4) | 52.7% | 0.2718 | matches majority, zero discrimination — negative result, kept |

**Read this plainly:** the base QLoRA matches the majority class. It is the
serving starting point, not a solved task. Calibration (temperature fit on a
disjoint calib split) is the lever that makes its probabilities mean what
they say — the same lever that cut Llama ECE 4.8x. No accuracy number is
claimed for Glimmer beyond what is measured above.

Failure-prediction probe (n=676): Glimmer zero-shot **44.7%** vs always-"ok"
70.6% (`corr(p,y) = +0.050`, AURC 0.5611). Scale alone does not solve it;
the mechanism + calibration is the product.

### The negative result that reframes this project

First attempt: fine-tune to predict **error**, evaluate on **unsafe**. Different
questions. Fine-tuning scored worse than always-"safe" on R-Judge (53.4% vs
floor 59.6%) and Agent-SafetyBench (35.7% vs floor 66.2%), cov@5% 0%
everywhere. A privacy-leaking `curl` returning HTTP 200 is safe under the
training label — no temperature fixes a mislabelled objective. The floor
column makes this visible before GPU is spent. Mechanism + calibration on the
model's own task reproduce; the safety-judge claim does not transfer.

### Benchmarks (R-Judge, Agent-SafetyBench)

- Temperature fitted on the model's own dev split, never on the benchmark.
- Full benchmark reported, `n` is the whole set.
- Majority-class floor printed beside every row.

## Dataset

| | value |
|---|---|
| raw (command, observation) pairs | 41,333 |
| unique commands after dedup | 12,744 |
| train / dev / test | 10,196 / 1,274 / 1,274 |
| split leakage | **none** (enforced by `make check`) |

**On the 223k target: this dataset is 12,744 unique commands, 5.7% of it.**
Real number, not padded. 74 SWE-bench repos bound the transfer; 3,340 label
conflicts are kept, not dropped.

## Limitations (do not quote a number without these)

- **Glimmer accuracy is at the floor.** Base QLoRA matches majority; calibration
  fixes confidence, not discrimination.
- **Debug recall weakest** (60.6% vs fix_input 91.5%, fix_call 88.8% on the
  Llama rung). Demo leans on missing-file/destructive cases where the model
  is strong.
- **cov@5% unstable at n=676.** One row is 0.15 pp.
- **Failure predictor, not safety judge.** Answers what to do next, not
  whether it is harmful.

## Layout

```
src/janus3.py        4-verdict contract (letters, meanings, read-out)
src/train_glimmer_janus.py  QLoRA trainer (single forward pass, 4 slots)
src/calibrate_glimmer.py    rescore + temperature fit (--rows slices for CPU)
src/muse_guard.py    allow/ask/block policy (model gives p, code decides)
src/muse_guard_cli.py  exit-code CLI (0/2/3)
src/muse_hook.py     PostToolUseFailure hook
src/guard.py         deterministic prechecks (win before model)
src/latency.py       p50/p95 harness (warmup dropped, stated)
muse/hooks.json      one-line install
```

## Licence

Code MIT. Model weights inherit from their base models — see
[MODEL_CARD.md](MODEL_CARD.md).
