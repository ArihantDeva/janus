#!/usr/bin/env bash
# Identical entrypoint for Amarel and GitHub; no live provider calls.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export JANUS_BACKEND=keyword OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${PYTHON:-python}"
"$PY" -c 'import torch, transformers, peft, pytest, numpy; print("model dependencies: present")'
case "${1:-hook}" in
  hook) targets=(tests/test_backends.py tests/test_janus_route.py tests/test_interface.py tests/test_hook_contract.py) ;;
  runner) targets=(tests/test_validation_runner.py) ;;
  remote) targets=(tests/test_remote_*.py tests/test_install_hook.py) ;;
  encoder) targets=(tests/test_encoder*.py tests/test_distill*.py tests/test_rlcd.py) ;;
  release) exec "$PY" -m pytest tests/ -q ;;
  legacy) export UV_PYTHON="$PY"; exec make check ;;
  mutations) exec "$PY" scripts/check_boundary_mutations.py ;;
  *) echo 'usage: check_boundaries.sh hook|runner|remote|encoder|release|legacy|mutations' >&2; exit 2 ;;
esac
# Full scope is the deterministic affected-target fallback: no cache to go stale.
exec "$PY" -m pytest "${targets[@]}" -q
