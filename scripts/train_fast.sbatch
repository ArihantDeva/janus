#!/bin/bash
#SBATCH --job-name=janus-fast
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ad2416/logs/fast_%j.out
#SBATCH --error=/scratch/ad2416/logs/fast_%j.err
#SBATCH --exclude=gpu017,gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018

# A DELIBERATELY SHORT run whose only purpose is to put a real trained checkpoint on
# disk quickly. The full array needs ~11h for its 3 seeds and ablations; this needs
# ~1h. Quality is not the goal here -- the goal is a working artifact to wire into
# Muse Code and to prove criterion 1 end to end, after which the long run's
# checkpoint replaces it with no code change (JANUS_ENCODER is a path).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== fast checkpoint run on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /scratch/ad2416/janussrc
/scratch/ad2416/venv-torch/bin/python src/train_distill.py \
  --seed 0 \
  --maxlen 1024 \
  --max-arg-chars 2000 \
  --epochs1 1 --epochs2 0.5 \
  --bs 16 \
  --out /scratch/ad2416/out_fast
