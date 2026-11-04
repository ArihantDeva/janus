#!/usr/bin/env bash
# Block until the allocation's socket exists. Bounded; never restarts anything.
set -euo pipefail
job="${1:?allocation id}"
run_dir="${JANUS_REMOTE_RUN_DIR:-/scratch/ad2416/janus-muse/run}"
deadline=$(( $(date +%s) + ${JANUS_REMOTE_READY_TIMEOUT:-300} ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  [ -f "$run_dir/encoder-$job.sock.ready" ] && exit 0
  case "$(squeue -h -j "$job" -o %T 2>/dev/null || true)" in
    RUNNING|PENDING) sleep 2 ;;
    *) echo "allocation $job never became ready" >&2; exit 3 ;;
  esac
done
echo "allocation $job readiness timed out" >&2
exit 4
