#!/bin/bash
#SBATCH --job-name=janus-glimmer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/ad2416/janus_glimmer_%j.out
# Full Glimmer Janus fine-tune: 21,901 train rows, 2,738 steps at bs=2/accum=4.
# At 10-20s/step the wall is 7.6-15.2h, so 16h with checkpoints every 50 steps:
# a timeout keeps everything, per checkpoint_due (last step always saves).
# Corpus MUST be on shared /scratch: compute nodes have their own /tmp.
# Shape check (50 rows) passes before this is ever submitted.
export JANUS_DATA=/scratch/ad2416/janusdata
export JANUS_CORPUS=/scratch/ad2416/janusdata/janus_distill.jsonl
export JANUS_OUT=/scratch/ad2416/janusout-glimmer-janus
export HF_HOME=/scratch/ad2416/hf
# Cache-only: the full Glimmer snapshot is pinned in HF_HOME (snapshot cec94f).
# Without these, transformers re-checks Hub metadata under anonymous rate
# limits and stalls at Fetching 0/2 (measured 2026-11-01, shape 63395319).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /scratch/ad2416/venv-glimmer/bin/python -u \
  /scratch/ad2416/janussrc/train_glimmer_janus.py \
  --seed 0 --epochs 1.0 --lr 3e-4 --grad-clip 1.0 --ckpt-every 50 --out "$JANUS_OUT"
