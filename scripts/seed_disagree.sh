#!/usr/bin/env bash
# Score all three clean-arm seed adapters on both benchmarks, keeping per-row
# probabilities so abstention rules can be tested WITHOUT a GPU.
#
# WHY: seed ensembles were refuted in run 15 for ACCURACY (mean-of-probs lost to
# the best single seed). That is a different question from ABSTENTION. A guard
# does not need the mean to be better; it needs a signal that says "these rows I
# should not answer". Seed disagreement is the standard proxy for epistemic
# uncertainty, and _confidence() in metrics.py cannot provide it because it only
# reads the model's own distance from 0.5 -- which is why cov@5 is structurally 0.
set -euo pipefail
SRC=/scratch/ad2416/janussrc
OUT=${1:-/scratch/ad2416/seedconf}
mkdir -p "$OUT"
for S in 0 1 2; do
  echo "=== seed $S ==="
  /scratch/ad2416/venv-torch/bin/python -u "$SRC/eval_bench.py" \
    --adapter /scratch/ad2416/janusout-clean/adapter_seed$S \
    --bench both --maxlen 512 --plain \
    --out "$OUT/bench_seed$S.json"
done
echo "SEED_CONF_DONE $OUT"
