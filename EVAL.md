# EVAL (frozen control, group-held-out)

Deployed checkpoint: reassembled from `weights/` + HuggingFace base
(see `weights/WEIGHTS.md`; original training path
`/scratch/ad2416/out_distill/model_freeze0_rlcd`, 4943273456B) +
`T=0.5276694558560848`. Eval is read-only; no tuning on eval.

## Distill

- agreement 85.337%
- ECE 0.171637

## RLCD

- agreement **82.199%** (beats frozen-control bar 82.20% → at bar, not above)
- ECE 0.102746
- per-class recall: 0.76923 / 0.92877 / 0.86346

## Baselines (same split)

- floor (majority): 55.19%
- keyword: 30.85%
- fast loop best RLCD: 73.37% / ECE 0.08071

## Two-stage abstention (I-CALM), local weights, 2026-11-02

Measured with the deployed local weights through the hook's own input path
(`src/eval_abstention.py`, dists in `eval/abstention_freeze0.json`,
`data/responses_v3/rule_held.jsonl`, n=1028, T=0.53). This is a local
re-measurement, not the frozen eval above; overall forced accuracy here is
80.2% against the headline 82.20% on the remote split — the headline does
not move.

At the operating threshold 0.60: coverage 90.2% (927/1028 answered),
answered accuracy 83.4%, forced best-guess on the 101 abstained rows 50.5%.
The gap is the product: rows the model refuses are the rows it would get
wrong half the time.

| threshold | coverage | answered acc | forced acc on abstained |
|---|---|---|---|
| 0.50 | 97.5% | 80.8% | 53.8% |
| 0.60 | 90.2% | 83.4% | 50.5% |
| 0.70 | 82.1% | 85.5% | 55.4% |
| 0.80 | 71.2% | 88.3% | 60.1% |
| 0.90 | 54.8% | 93.6% | 63.9% |

Curve reported, threshold not tuned on this set (standing rule). Tier-2
accuracy on the abstained rows is deliberately not proxied: it is only
measurable with frontier calls (`eval_router.py`, unrun).

## Honest reading

Frozen control is good enough to ship as-is; it does not beat the bar with
margin. Calibration (ECE 0.10) is the stronger story than raw agreement.

### Seed variance (second-seed confirm, done)

Non-frozen RLCD agreement across seeds on the same split (`result_seed*.json`):

| seed | acc | ece |
|---|---|---|
| 0 | 0.6701 | 0.1109 |
| 1 | 0.3894 | 0.2247 |
| 2 | 0.5257 | 0.1062 |
| freeze0 (deployed) | **0.8220** | 0.1027 |

The freeze is the finding, not a fallback: non-frozen training is unstable
on this split (38.9-67.0, far wider than the earlier ±14pp estimate), and the
frozen encoder wins on every seed. Freezing the encoder is load-bearing —
this is why the deployed checkpoint is `model_freeze0_rlcd`.
