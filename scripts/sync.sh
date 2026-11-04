#!/usr/bin/env bash
# Push code + data to Amarel. Everything lands on /scratch: the home
# directory has a hard ~2GB quota and will silently evict large files.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${JANUS_HOST:-amarel}"
SCRATCH="${JANUS_SCRATCH:-/scratch/ad2416}"

echo "== syncing code -> $HOST:$SCRATCH/janussrc =="
ssh "$HOST" "mkdir -p $SCRATCH/janussrc $SCRATCH/janusdata"
rsync -az src/ "$HOST:$SCRATCH/janussrc/"

echo "== syncing data -> $HOST:$SCRATCH/janusdata =="
rsync -az data/swe_*.jsonl data/noul_*.jsonl data/rjudge.jsonl "$HOST:$SCRATCH/janusdata/"

echo "== remote state =="
ssh "$HOST" "wc -l $SCRATCH/janusdata/*.jsonl"
