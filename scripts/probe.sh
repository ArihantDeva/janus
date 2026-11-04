#!/usr/bin/env bash
# Submit the Glimmer 30B zero-shot probe.
#
# Node selection is the whole trick here. The gpuk* nodes mount a DIFFERENT
# /scratch (/scache, 257T) that the login node cannot see, so a job there
# writes its log into a void and looks like it produced no output. Every
# allocation below therefore excludes gpuk*; gpu*/volta* share the login
# filesystem.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${JANUS_HOST:-amarel}"
SCRATCH="${JANUS_SCRATCH:-/scratch/ad2416}"
N="${JANUS_N:-676}"

ssh "$HOST" "cat > $SCRATCH/run_probe.sh <<'SBATCH'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=gprobe
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=$SCRATCH/gprobe.out
export JANUS_DATA=$SCRATCH/janusdata
export HF_HOME=$SCRATCH/hf
exec $SCRATCH/venv-glimmer/bin/python -u $SCRATCH/janussrc/glimmer_probe.py --n $N --load-4bit --dump $SCRATCH/gl_zero.npz
SBATCH
chmod +x $SCRATCH/run_probe.sh && sbatch --parsable $SCRATCH/run_probe.sh"
