#!/usr/bin/env bash
# Correctness gate: runs automatically after every passing benchmark in
# run_experiment. Any failure => checks_failed (no keep), same as a crash.
# Suppressed to errors only: only the tail reaches the agent on failure.
set -euo pipefail
cd "$(dirname "$0")/.."

# The validation that matters for training-loop changes: encoder/distill/RLCD
# semantics + artifact + boundary gates. Full make check runs separately on
# Amarel via validate_amarel.sbatch (it needs the cluster for model scopes).
UV_PY="$(mise exec -- which uv 2>/dev/null || echo uv)"
SCRIPTS_CHANGED="$(git diff --name-only HEAD -- scripts/ | head -5 || true)"
if [ -n "$SCRIPTS_CHANGED" ]; then
  echo "scripts changed; running boundary mutation gate is Amarel-only, deferring" >&2
fi
mise exec -- python3 -m pytest tests/test_encoder.py tests/test_distill.py \
  tests/test_rlcd.py tests/test_distill_artifacts.py tests/test_choice_codes.py -q 2>&1 | tail -8
