#!/bin/bash
#SBATCH --job-name=janus-shape
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ad2416/janus_shape_%j.out
# SHAPE: new hard-target + class-weight recipe on a 500-row slice.
# Weak GPU fine. Proves loss descends and no class collapses before L40S.
export JANUS_DATA=/scratch/ad2416/janusdata
export JANUS_CORPUS=/scratch/ad2416/janusdata/janus_distill.jsonl
export JANUS_OUT=/scratch/ad2416/janusshape-hard
export HF_HOME=/scratch/ad2416/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /scratch/ad2416/venv-glimmer/bin/python -u \
  /scratch/ad2416/janussrc/train_glimmer_janus.py \
  --seed 0 --epochs 2 --lr 3e-4 --grad-clip 1.0 --max-rows 500 --ckpt-every 50 --out "$JANUS_OUT"
