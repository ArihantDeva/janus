#!/usr/bin/env bash
# One route to install, verify or disable the encoder hook in a Muse workspace.
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:-verify}"
workspace="${2:-$PWD}"
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
import json, install_muse_hook as I
result = getattr(I, '$action')('$workspace')
print(json.dumps(result if isinstance(result, dict) else {'path': result}, indent=2))
"
