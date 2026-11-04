#!/bin/bash
# One command, raw session stores -> scored checkpoint.
#
#   ./scripts/pipeline.sh <variant> [seed]
#
# variant: base | seed<N> | stage1 | freeze | temp1 | conf06 | long
# Runs the same code the cluster runs, in the same order, with the same split.
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT="${1:-base}"; SEED="${2:-0}"
case "$VARIANT" in
  stage1) EXTRA="--rlcd-strength 0 --epochs2 0" ;;
  freeze) EXTRA="--freeze-encoder --epochs2 0" ;;
  temp1)  EXTRA="--temp 1.0 --epochs2 0" ;;
  conf06) EXTRA="--min-conf 0.6 --epochs2 0" ;;
  long)   EXTRA="--maxlen 8192 --epochs2 0" ;;
  *)      EXTRA="" ;;
esac
if [ ! -f data/distill/janus.jsonl ]; then
  echo "== harvest ==";  uv run python src/build_distill.py
  echo "== label ==";    uv run python src/label_janus.py
fi
echo "== train ($VARIANT seed $SEED) =="
uv run --with torch --with transformers --with peft --with numpy \
  python src/train_distill.py --seed "$SEED" $EXTRA \
  --bs 8 --max-arg-chars 4000 --epochs1 3 --out out_distill
echo "== eval =="
uv run --with torch --with transformers --with numpy \
  python src/eval_distill.py --model "out_distill/model_seed${SEED}_distill" \
  --split-seed "$SEED" --out "out_distill/eval_seed${SEED}.json"
