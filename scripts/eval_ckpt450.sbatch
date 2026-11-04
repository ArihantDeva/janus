#!/bin/bash
#SBATCH --job-name=janus-eval450
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ad2416/janus_eval450_%j.out
# EVAL-ONLY: score the pre-flatline ckpt (step 450) on the full n=3410 EVAL.
# Minutes, not hours: no training, just zero-shot + ckpt forward passes.
export JANUS_DATA=/scratch/ad2416/janusdata
export JANUS_CORPUS=/scratch/ad2416/janusdata/janus_distill.jsonl
export JANUS_OUT=/scratch/ad2416/januseval-450
export HF_HOME=/scratch/ad2416/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /scratch/ad2416/venv-glimmer/bin/python -u \
  /scratch/ad2416/janussrc/train_glimmer_janus.py \
  --seed 0 --eval-only \
  --resume-from /scratch/ad2416/janusout-glimmer-janus/janus_ckpt_seed0_step450 \
  --out "$JANUS_OUT"
