#!/usr/bin/env bash
# One owned classifier allocation. Wall-limited by Slurm; no restart logic here.
#SBATCH --job-name=janus-muse-encoder
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ad2416/janus-muse/logs/service_%j.out
#SBATCH --error=/scratch/ad2416/janus-muse/logs/service_%j.err
set -euo pipefail
cd /scratch/ad2416/janus-muse
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export JANUS_BACKEND=encoder JANUS_ENCODER_TRANSPORT=local
# Frozen-control RLCD checkpoint with its saved calibration; never retrained here.
export JANUS_ENCODER=/scratch/ad2416/out_distill/model_freeze0_rlcd
export JANUS_ENCODER_HEAD=/scratch/ad2416/out_distill/model_freeze0_rlcd/head.pt
export JANUS_ENCODER_TEMPERATURE=/scratch/ad2416/out_distill/temperature_freeze0.json
export JANUS_REMOTE_RUN_DIR=/scratch/ad2416/janus-muse/run
mkdir -p "$JANUS_REMOTE_RUN_DIR"
exec /scratch/ad2416/venv-torch/bin/python src/remote_service.py
