#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=janussafety
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=/scratch/ad2416/janussafety.out
# Fine-tune Llama-3.2-1B on the SAFETY corpora and score both evals.
#
# 1 epoch, not 3. Measured 0.44 s/step over 15,139 steps is 1.85h at 3 epochs'
# step count; 3 epochs on this corpus would be ~5.5h and the first attempt was
# projected to overrun a 2h wall clock. 1 epoch over 20k rows is already more
# gradient steps than the 3 epochs of 6.7k rows that produced the 1B rung, and a
# wall clock is not the place to discover you wanted fewer epochs.
#
# JANUS_CORPUS=safety is the load-bearing line. The previous run trained on
# "will this command error" and was scored on "is this action unsafe", which is
# why its numbers sat below the majority-class floor. The corpus selects the
# question; the evals must match it.
export JANUS_DATA=/scratch/ad2416/janusdata
export JANUS_OUT=/scratch/ad2416/janusout-safety
export JANUS_CORPUS=safety
export HF_HOME=/scratch/ad2416/hf
cd /scratch/ad2416 || exit 1

/scratch/ad2416/venv-torch/bin/python -u /scratch/ad2416/janussrc/train_lora.py \
  --seed 0 --epochs 1 --bs 4 --lr 2e-4 --maxlen 256 \
  --out "$JANUS_OUT" || exit 2

echo "TRAIN_DONE"
