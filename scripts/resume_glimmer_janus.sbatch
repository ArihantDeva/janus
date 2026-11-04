#!/bin/bash
#SBATCH --job-name=janus-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/ad2416/janus_resume_%j.out
# RESUME: from the last pre-spike ckpt (step 200, loss 0.0023) with a
# quieter optimizer: effective batch 32 (bs2 x accum16), lr 1e-4.
# Continues the data cycle (resume-steps) instead of replaying the head.
export JANUS_DATA=/scratch/ad2416/janusdata
export JANUS_CORPUS=/scratch/ad2416/janusdata/janus_distill.jsonl
export JANUS_OUT=/scratch/ad2416/janusout-glimmer-resume
export HF_HOME=/scratch/ad2416/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /scratch/ad2416/venv-glimmer/bin/python -u \
  /scratch/ad2416/janussrc/train_glimmer_janus.py \
  --seed 0 --epochs 1.0 --lr 1e-4 --accum 8 --grad-clip 1.0 \
  --resume-from /scratch/ad2416/janusout-glimmer-janus/janus_ckpt_seed0_step200 \
  --resume-steps 200 --ckpt-every 50 --out "$JANUS_OUT"
