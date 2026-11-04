#!/bin/bash
#SBATCH --job-name=janus-distill
#SBATCH --partition=gpu
#SBATCH --requeue
# NO hardware constraint: constraining to ampere|adalovelace narrowed capacity until
# the job sat in Priority for 20+ minutes. Instead the CODE adapts -- bf16 on sm80+,
# fp32 on sm70 -- and the wall clock is generous (20h) because fp32 on a V100 is ~3x
# slower per step. Correctness over speed: fp16 produced a NaN seed twice.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/ad2416/logs/dist_%A_%a.out
#SBATCH --error=/scratch/ad2416/logs/dist_%A_%a.err
#SBATCH --exclude=gpu017,gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018

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
  $ARGS --bs 8 --max-arg-chars 4000 \
  --epochs1 2 --epochs2 1 --out /scratch/ad2416/out_distill
# epochs 3+2 was measured at 12.2h (stage 1) + 8.1h (stage 2) = 20.3h on the fp32
# path, against a 20h wall: it would have been killed 18 min from the end. 2+1 is
# 8.1h + 4.1h = 12.2h, inside the wall with ~8h of margin, and the loss was already
# 0.28 at 0.33 epochs so the extra passes were not buying convergence.
