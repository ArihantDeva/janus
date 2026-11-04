#!/bin/bash
#SBATCH --job-name=janus-enc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ad2416/logs/enc_%A_%a.out
#SBATCH --error=/scratch/ad2416/logs/enc_%A_%a.err
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018

# One array task per seed. The seeds matter: with an unshuffled loop every seed
# saw the identical row order and n=3 measured init variance only (run 29).
set -euo pipefail
SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "=== seed $SEED on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /scratch/ad2416/janussrc
/scratch/ad2416/venv-torch/bin/python src/encoder_llama.py \
  --seed "$SEED" \
  --maxlen 2048 \
  --epochs 5 \
  --lr 1e-3 \
  --head-lr 1e-3 \
  --bs 8 \
  --lora-r 16 \
  --out /scratch/ad2416/out_encoder
