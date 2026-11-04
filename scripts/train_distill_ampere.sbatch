#!/bin/bash
#SBATCH --job-name=janus-distill-amp
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --constraint=ampere|adalovelace
# Ampere/Ada ONLY. This is submitted alongside the unrestricted variant rather than
# instead of it: the code adapts to either (bf16 on sm80+, fp32 on sm70), so whichever
# hardware becomes available first produces a valid result, and the fp32 path on a
# V100 is ~3x slower. Requesting the fast hardware and also running on the slow
# hardware is strictly better than waiting for one of them.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/ad2416/logs/dist_%A_%a.out
#SBATCH --error=/scratch/ad2416/logs/dist_%A_%a.err
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018

# One array task per variant. Memory discipline: everything is encoded per step,
# never as one whole-corpus tensor (that OOM'd a 32GB GPU on seed 1).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
VARIANT=${SLURM_ARRAY_TASK_ID:-0}
echo "=== variant $VARIANT on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /scratch/ad2416/janussrc

case "$VARIANT" in
  0) ARGS="--seed 0" ;;
  1) ARGS="--seed 1" ;;
  2) ARGS="--seed 2" ;;
  3) ARGS="--seed 0 --rlcd-strength 0 --epochs2 0" ;;
  4) ARGS="--seed 0 --freeze-encoder" ;;
  5) ARGS="--seed 0 --temp 1.0 --epochs2 0" ;;
  6) ARGS="--seed 0 --min-conf 0.6 --epochs2 0" ;;
  7) ARGS="--seed 0 --maxlen 8192 --epochs2 0" ;;
esac

/scratch/ad2416/venv-torch/bin/python src/train_distill.py \
  $ARGS --bs 8 --max-arg-chars 4000 --epochs1 3 --out /scratch/ad2416/out_distill_amp
