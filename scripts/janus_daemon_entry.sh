#!/bin/bash
# Warm Janus daemon: local encoder weights, uv env (system python has no torch).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/Users/arihantdeva/.local/bin:/usr/bin:/bin:/opt/homebrew/bin"
export JANUS_BACKEND=janus
export JANUS_ENCODER_TRANSPORT=local
export JANUS_ENCODER="$PWD/weights/model_freeze0_rlcd"
export JANUS_ENCODER_HEAD="$PWD/weights/model_freeze0_rlcd/head.pt"
export JANUS_THRESHOLD="${JANUS_THRESHOLD:-0.60}"
export PYTHONUNBUFFERED=1
exec mise exec -- uv run --with numpy --with torch --with transformers \
  --with peft --with safetensors python src/janus_daemon.py serve
