#!/usr/bin/env bash
# Runs INSIDE the allocation: the socket is on the node's own filesystem.
# srun starts in $HOME, so the source path must be absolute.
set -euo pipefail
root="${JANUS_REMOTE_DIR:-/scratch/ad2416/janus-muse}"
job="${1:?allocation id}"
export JANUS_REMOTE_JOB="$job"
export JANUS_REMOTE_RUN_DIR="${JANUS_REMOTE_RUN_DIR:-$root/run}"
cd "$root"
exec /scratch/ad2416/venv-torch/bin/python "$root/src/remote_client.py"
