#!/usr/bin/env bash
# One AutoResearch iteration: sync code, submit one SLURM train+eval job on
# Amarel, wait, emit METRIC lines. Slow workload (8-12h) — single runs.
set -euo pipefail

HOST="${JANUS_HOST:-amarel}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes)
acct() { ssh "${SSH_OPTS[@]}" "$HOST" "$@" 2>/dev/null | tr -d '\r' || true; }
LOOP_SRC=/scratch/ad2416/janussrc-loop
LOOP_OUT=/scratch/ad2416/out_loop
POLL="${POLL_SECONDS:-60}"
MAXWAIT="${MAX_WAIT_SECONDS:-75600}"

# --- hyperparameters (env; baseline = frozen control reproduction) ------------
SEED="${SEED:-0}"
FREEZE="${FREEZE:-1}"
EPOCHS1="${EPOCHS1:-2}"
EPOCHS2="${EPOCHS2:-1}"
RLCD_STRENGTH="${RLCD_STRENGTH:-0.5}"
TEMP="${TEMP:-2.0}"
MIN_CONF="${MIN_CONF:-0.0}"
HEAD_LR="${HEAD_LR:-0.001}"
LORA_R="${LORA_R:-16}"
MAXLEN="${MAXLEN:-2048}"
MAX_ARG_CHARS="${MAX_ARG_CHARS:-4000}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- fast pre-check: never burn an allocation on a typo ----------------------
python3 - src/train_distill.py src/rlcd.py src/calibrate.py src/janus_interface.py <<'PY' || { echo "METRIC rlcd_acc=0"; echo "PRECHECK_FAILED"; exit 1; }
import pathlib, sys
root = pathlib.Path("src")
for f in sys.argv[1:]:
    p = pathlib.Path(f)
    if not p.exists():
        print(f"missing {p}"); sys.exit(1)
    compile(p.read_text(), str(p), "exec")
print("precheck ok")
PY

# --- instrument identity: the cluster must run THIS repo's code ---------------
LOCAL_MD5=$(md5 -q src/train_distill.py src/rlcd.py src/calibrate.py src/janus_interface.py 2>/dev/null | tr '\n' ' ')
acct "mkdir -p $LOOP_SRC $LOOP_OUT /scratch/ad2416/logs"
rsync -az --delete src/ "$HOST:$LOOP_SRC/src/"
rsync -az scripts/train_distill.sbatch "$HOST:$LOOP_SRC/train_distill.sbatch.ref"
CLUSTER_MD5=$(acct "cd $LOOP_SRC && md5sum src/train_distill.py src/rlcd.py src/calibrate.py src/janus_interface.py 2>/dev/null | cut -d' ' -f1 | tr '\n' ' '")
if [ -n "$CLUSTER_MD5" ] && [ "$LOCAL_MD5" != "$CLUSTER_MD5" ]; then
  echo "METRIC rlcd_acc=0"
  echo "INSTRUMENT_DRIFT local='$LOCAL_MD5' cluster='$CLUSTER_MD5'"
  echo "refusing to run: cluster copy differs from this repo"
  exit 1
fi

# --- attach to an existing job (recovery path) --------------------------------
JID=""
if [ "${1:-}" = "--jid" ]; then
  JID="${2:-}"
  [ -n "$JID" ] || { echo "METRIC rlcd_acc=0"; echo "--jid needs a job id"; exit 1; }
  echo "attaching to existing job $JID" >&2
fi

# --- submit ------------------------------------------------------------------
if [ -z "$JID" ]; then
RUN_NAME="loop_$(date +%Y%m%d_%H%M%S)${OUT_SUFFIX:+_$OUT_SUFFIX}"
FREEZE_CLI=""; [ "$FREEZE" = "1" ] && FREEZE_CLI="--freeze-encoder"
WRAP="/scratch/ad2416/venv-torch/bin/python src/train_distill.py --seed $SEED $FREEZE_CLI --lora-r $LORA_R --epochs1 $EPOCHS1 --epochs2 $EPOCHS2 --rlcd-strength $RLCD_STRENGTH --temp $TEMP --min-conf $MIN_CONF --head-lr $HEAD_LR --maxlen $MAXLEN --max-arg-chars $MAX_ARG_CHARS $EXTRA_ARGS --bs 8 --corpus /scratch/ad2416/janussrc/data/distill/janus.jsonl --out $LOOP_OUT/$RUN_NAME"
JID=$(acct "cd $LOOP_SRC && sbatch --parsable --job-name=janus-loop --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=20:00:00 --output=/scratch/ad2416/logs/loop_%A.out --error=/scratch/ad2416/logs/loop_%A.err --exclude=gpu017,gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018 --wrap='$WRAP'")
case "$JID" in
  ''|*[!0-9]*) echo "METRIC rlcd_acc=0"; echo "SUBMIT_FAILED jid='$JID'"; exit 1 ;;
esac
echo "submitted $JID run=$RUN_NAME" >&2
echo "$JID $RUN_NAME" > /tmp/janus_loop_last_jid
fi

# --- wait (bounded, resilient to blips) ---------------------------------------
elapsed=0
fails=0
while :; do
  state=$(acct "sacct -j $JID -o State -n | head -1" | tr -d ' \r')
  if [ -z "$state" ]; then
    fails=$((fails + 1))
    if [ "$fails" -ge 12 ]; then state=CONNECTIVITY_LOST; fi
  else
    fails=0
    case "$state" in
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY) break ;;
    esac
  fi
  if [ "$elapsed" -ge "$MAXWAIT" ]; then
    echo "METRIC rlcd_acc=0"; echo "TIMEOUT after ${elapsed}s job=$JID"; exit 1
  fi
  sleep "$POLL"
  elapsed=$((elapsed + POLL))
done

tail_log=$(acct "tr '\r' '\n' < /scratch/ad2416/logs/loop_${JID}.out 2>/dev/null | tail -120")

if [ "$state" = "CONNECTIVITY_LOST" ]; then
  echo "METRIC rlcd_acc=0"
  echo "CONNECTIVITY_LOST job=$JID — recover with: ./.auto/measure.sh --jid $JID"
  exit 1
fi

if ! grep -q "wrote .*result_.*\.json" <<<"$tail_log"; then
  echo "METRIC rlcd_acc=0"
  echo "RUN_INCOMPLETE state=$state job=$JID"
  grep -E "FAILED|Error|Traceback|SystemExit|non-finite" <<<"$tail_log" | tail -8 || true
  exit 1
fi

# --- extract metrics from the result JSON --------------------------------------
RUN_DIR_NAME=$(acct "ls -dt $LOOP_OUT/loop_* 2>/dev/null | head -1")
acct "cat $RUN_DIR_NAME/result_*.json" | python3 - "$JID" "$state" <<'PY'
import json, re, sys
jid, state = sys.argv[1], sys.argv[2]
raw = sys.stdin.read()
m = re.search(r"\{.*\}", raw, re.S)
if not m:
    print("METRIC rlcd_acc=0"); print("RESULT_PARSE_FAILED"); sys.exit(1)
d = json.loads(m.group(0))
r = d["results"]["rlcd"]; s = d["results"]["distill"]
def emit(k, v): print(f"METRIC {k}={v:.5f}")
emit("rlcd_acc", r["acc"]); emit("distill_acc", s["acc"])
emit("rlcd_ece", r["ece"]); emit("distill_ece", s["ece"]); emit("kl", r["kl"])
emit("per_debug", r["per"]["debug"][1] or 0)
emit("per_fix_input", r["per"]["fix_input"][1] or 0)
emit("per_fix_call", r["per"]["fix_call"][1] or 0)
emit("floor", d["tool_only_floor"])
emit("margin", r["acc"] - d["tool_only_floor"])
emit("ood_acc", d["results"].get("ood", {}).get("acc", 0))
emit("ood_n", d["results"].get("ood", {}).get("n", 0))
emit("n_train", d["n_train"]); emit("n_eval", d["n_eval"])
emit("nonfinite_steps", r.get("nonfinite_steps", 0))
print(f"RUN_OK job={jid} state={state} seed={d['seed']} "
      f"eval_acc_rlcd={r['acc']:.4f} margin={r['acc']-d['tool_only_floor']:+.4f}")
PY
