#!/usr/bin/env bash
#SBATCH --job-name=janus-boundaries
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ad2416/logs/boundaries_%j.out
#SBATCH --error=/scratch/ad2416/logs/boundaries_%j.err
set -euo pipefail
cd "${JANUS_VALIDATION_DIR:?use an isolated snapshot}"
# Separate pytest overlay; never install into the active training environment.
export PYTHONPATH=/scratch/ad2416/venv-torch/lib/python3.9/site-packages
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JANUS_BACKEND=keyword
export PYTHON=/scratch/ad2416/venv-torch/bin/python
# The scope loop lives in run_scopes.sh, which keeps errexit in the shell that
# runs the scopes. Inlining it in `bash -c` silently dropped the guard.
exec /home/ad2416/.local/bin/uv run --no-project --python "$PYTHON" \
  --with pytest==8.4.2 bash scripts/run_scopes.sh scripts/check_boundaries.sh "${@:-hook}"
