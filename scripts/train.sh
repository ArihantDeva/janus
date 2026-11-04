#!/usr/bin/env bash
# Submit the 1B LoRA rung for three seeds.
#
# Amarel gotchas encoded here:
#   --cpus-per-task=8 --mem=32G  default is 1GB/CPU and the job dies at ~485MB
#   JANUS_DATA/JANUS_OUT on /scratch  home quota is ~2GB and evicts silently
#   python -u                     stdout is buffered when redirected
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${JANUS_HOST:-amarel}"
SCRATCH="${JANUS_SCRATCH:-/scratch/ad2416}"
EPOCHS="${JANUS_EPOCHS:-2}"

for seed in 0 1 2; do
  ssh "$HOST" "sbatch --parsable --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G --time=02:30:00 --job-name=noul$seed \
    --output=$SCRATCH/januslab-lora$seed.out \
    --wrap='JANUS_DATA=$SCRATCH/janusdata JANUS_OUT=$SCRATCH/janusout HF_HOME=$SCRATCH/hf \
      $SCRATCH/venv-torch/bin/python -u $SCRATCH/janussrc/train_lora.py \
      --seed $seed --epochs $EPOCHS'"
done
