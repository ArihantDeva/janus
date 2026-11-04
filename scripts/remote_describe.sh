#!/usr/bin/env bash
# Report which checkpoint/calibration the owned allocation is actually serving.
# Runs INSIDE the allocation, so the unix socket is on the node's own filesystem.
set -euo pipefail
root="${JANUS_REMOTE_DIR:-/scratch/ad2416/janus-muse}"
job="${1:?allocation id}"
export JANUS_REMOTE_JOB="$job"
export JANUS_REMOTE_RUN_DIR="${JANUS_REMOTE_RUN_DIR:-$root/run}"
exec /scratch/ad2416/venv-torch/bin/python "$root/src/remote_describe.py"
