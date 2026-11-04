#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=janusloop
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=/scratch/ad2416/loop.out
#
# One AutoResearch iteration, end to end: train on the safety corpora, then score
# BOTH benchmarks and the memorisation gap. Everything in a single allocation so
# an iteration is one wall clock, not three.
#
# Hyperparameters arrive as environment variables from .auto/measure.sh. They are
# read here rather than hardcoded so an iteration needs no file edit -- the loop
# changes values, not code, and a crashed run cannot leave the script modified.
#
# The benchmarks are EVAL ONLY. eval_bench.py never reads them for training and
# nothing here feeds them back into the corpus.
set -uo pipefail

# Data dir is overridable so a corpus variant can be evaluated without
# editing this file. Defaults to the baseline corpus.
export JANUS_DATA="${JANUS_DATA:-/scratch/ad2416/janusdata}"
export JANUS_OUT="${JANUS_OUT:-/scratch/ad2416/loopout}"
export JANUS_CORPUS=safety
export HF_HOME=/scratch/ad2416/hf
cd /scratch/ad2416 || exit 90

PY=/scratch/ad2416/venv-torch/bin/python
mkdir -p "$JANUS_OUT"

echo "=== ITERATION config ==="
echo "  EPOCHS=${EPOCHS:-1} LR=${LR:-2e-4} BS=${BS:-4} MAXLEN=${MAXLEN:-512} SEED=${SEED:-0}"
echo "  PRIOR_SHIFT=${PRIOR_SHIFT:-0}"
echo "  LORA_R=${LORA_R:-16} LORA_ALPHA=${LORA_ALPHA:-32} LORA_DROPOUT=${LORA_DROPOUT:-0.05} TARGETS=${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

# SKIP_TRAIN=1 reuses the adapter already on disk and only re-scores. A bug in
# the EVAL wiped out a completed GPU-hour because the only way to retest was to
# retrain identical hyperparameters from scratch. Eval-code changes must never
# cost a training run.
if [ "${SKIP_TRAIN:-0}" = "1" ] && [ -d "$JANUS_OUT/adapter_seed${SEED:-0}" ]; then
  echo "SKIP_TRAIN=1: reusing $JANUS_OUT/adapter_seed${SEED:-0}"
else
$PY -u /scratch/ad2416/janussrc/train_lora.py \
  --epochs "${EPOCHS:-1}" --lr "${LR:-2e-4}" --bs "${BS:-4}" \
  --maxlen "${MAXLEN:-512}" --seed "${SEED:-0}" \
  --lora-r "${LORA_R:-16}" --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --target-modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}" \
  --class-weight "${CLASS_WEIGHT:-none}" \
  --model "${JANUS_BASE_MODEL:-unsloth/Llama-3.2-1B}" \
  --task "${JANUS_TASK:-noul}" \
  --out "$JANUS_OUT" || { echo "TRAIN_FAILED"; exit 91; }
fi

echo "=== EVALS ==="
$PY -u /scratch/ad2416/janussrc/eval_bench.py \
  --adapter "$JANUS_OUT/adapter_seed${SEED:-0}" \
  --bench both \
  --maxlen "${MAXLEN:-512}" \
  --prior-shift "${PRIOR_SHIFT:-0}" \
  $([ "${PLAIN:-0}" = "1" ] && echo --plain) \
  --out "$JANUS_OUT/bench.json" || { echo "EVAL_FAILED"; exit 92; }

echo "LOOP_DONE"
