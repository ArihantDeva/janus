#!/bin/bash
# A/B capture A: WITHOUT Janus hook. Disables the installed hook by moving
# .muse/hooks.json aside (Muse hook subprocesses do not inherit our exported
# JANUS_HOOK_DISABLE), restores it on exit. Real session, no advisory possible.
set -euo pipefail
cd "$(dirname "$0")/../.."
mv .muse/hooks.json /tmp/hooks.json.disabled-by-ab
trap 'mv /tmp/hooks.json.disabled-by-ab .muse/hooks.json' EXIT
export TERM=xterm-256color
echo "=== WITHOUT Janus: same failure, no advisory possible (hook uninstalled) ==="
muse exec --yolo --trust-workspace "Run: cat /nonexistent/x.txt Then say which response class and which classifier name advised you."
echo
echo "=== exit $? ==="
