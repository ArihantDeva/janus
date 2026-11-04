#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=glimlora
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=/scratch/ad2416/glimlora.out
# 1 epoch over 5412 rows at bs=2 x accum=4 is ~677 steps at ~20s/step, so the
# projected runtime is ~3.8h. The wall clock is 8h and checkpoints land every 50
# steps: the first attempt used 4h with a save ONLY on the final step, which
# would have timed out ~3 minutes before saving and destroyed the run.
export JANUS_DATA=/scratch/ad2416/janusdata
export JANUS_OUT=/scratch/ad2416/janusout-glimmer
export HF_HOME=/scratch/ad2416/hf
exec /scratch/ad2416/venv-glimmer/bin/python -u /scratch/ad2416/janussrc/train_glimmer.py --seed 0 --epochs 1 --ckpt-every 50
