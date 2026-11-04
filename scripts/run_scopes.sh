#!/usr/bin/env bash
# Ordered validation scopes. A failing scope must stop the run: a later green
# scope is not evidence that an earlier one passed.
# Usage: run_scopes.sh RUNNER [SCOPE...]   (RUNNER is a script; the scope is $1)
set -euo pipefail
runner="${1:?usage: run_scopes.sh RUNNER [SCOPE...]}"
shift
[ "$#" -gt 0 ] || set -- hook
# The launcher's overlay interpreter carries pytest; the inherited PYTHON does not.
export PYTHON="$(command -v python)"
for scope in "$@"; do
  bash "$runner" "$scope"
done
