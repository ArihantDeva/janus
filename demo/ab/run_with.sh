#!/bin/bash
# A/B capture B: WITH Janus hook. Real muse session, hook enabled (warm daemon).
# Run inside: mise exec -- uvx --from asciinema asciinema rec demo/ab/with.cast -c ./demo/ab/run_with.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
unset JANUS_HOOK_DISABLE
export TERM=xterm-256color
echo "=== WITH Janus: same failure, hook advisory expected ==="
muse exec --yolo --trust-workspace "Run: cat /nonexistent/x.txt Then say which response class and which classifier name advised you."
echo
echo "=== exit $? ==="
