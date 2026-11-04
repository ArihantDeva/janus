#!/bin/bash
# Run the full distillation locally on Apple MPS. Identical script, identical data,
# identical split seed as the cluster run -- so whichever finishes first is a real
# result, not a reduced one. MPS is slower than an A100 but it does not queue.
set -uo pipefail
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
cd "$HOME/jevlab"
LOG=logs/local_distill_${1:-0}.log
mkdir -p logs
uv run --with torch --with transformers --with peft --with numpy \
  python src/train_distill.py \
  --seed "${1:-0}" \
  --bs 4 \
  --max-arg-chars 4000 \
  --epochs1 3 \
  --out out_distill_local > "$LOG" 2>&1
