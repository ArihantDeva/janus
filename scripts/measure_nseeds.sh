#!/usr/bin/env bash
# Evaluate the same configuration across several seeds and aggregate.
#
# The noise floor measured in this project is sd ~0.02 AUC from a pure seed
# change, which is larger than nearly every effect this session chased. A
# single-seed number for ASB is therefore not resolvable, which is exactly why
# no corpus/task-shape conclusion can be drawn from one. This runs the identical
# configuration at each seed and reports the mean and sd per benchmark.
#
# Nothing here is tuned per benchmark or per seed: same config, only SEED varies.
set -uo pipefail
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2}"
export SKIP_TRAIN=1 PLAIN=0 MAXLEN=512 LR=2e-4 LORA_R=16
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for s in $SEEDS; do
  echo "=== SEED $s ==="
  SEED=$s ./.auto/measure.sh >"$TMP/s$s.log" 2>&1 || {
      echo "seed $s FAILED"; tail -5 "$TMP/s$s.log"; exit 1; }
  rg --no-ignore -o 'METRIC [a-z0-9_]+=[0-9.]+' "$TMP/s$s.log" | sed "s/^/seed$s /"
done

python3 - "$TMP" $SEEDS <<'PY'
import sys, pathlib, re, statistics as st
tmp, seeds = pathlib.Path(sys.argv[1]), sys.argv[2:]
want = ["acc_rjudge","acc_asb","auc_rjudge","auc_asb","aurc_rjudge","aurc_asb",
        "floor_rjudge","floor_asb","ece_rjudge","ece_asb"]
data = {}
for s in seeds:
    d = {}
    for m in re.finditer(r"METRIC ([a-z0-9_]+)=([0-9.]+)", (tmp/f"s{s}.log").read_text()):
        d[m.group(1)] = float(m.group(2))
    data[s] = d
print()
print(f"{'metric':14s} {'mean':>8s} {'sd':>7s} {'min':>8s} {'max':>8s}   values")
agg = {}
for k in want:
    v = [data[s][k] for s in seeds if k in data[s]]
    if not v: continue
    sd = st.stdev(v) if len(v) > 1 else 0.0
    agg[k] = (st.mean(v), sd)
    print(f"{k:14s} {st.mean(v):8.4f} {sd:7.4f} {min(v):8.4f} {max(v):8.4f}   "
          + " ".join(f"{x:.3f}" for x in v))
print()
am  = st.mean([ (data[s]["aurc_rjudge"]+data[s]["aurc_asb"])/2 for s in seeds ])
print(f"METRIC aurc_mean={am:.5f}")
mj = agg["acc_rjudge"]; ma = agg["acc_asb"]
print(f"R-JUDGE acc {mj[0]*100:.1f}% +- {mj[1]*100:.1f} ptsd vs floor {agg['floor_rjudge'][0]*100:.1f}%"
      f"  lift {100*(mj[0]-agg['floor_rjudge'][0]):+.1f}pt")
print(f"ASB     acc {ma[0]*100:.1f}% +- {ma[1]*100:.1f} ptsd vs floor {agg['floor_asb'][0]*100:.1f}%"
      f"  lift {100*(ma[0]-agg['floor_asb'][0]):+.1f}pt")
# Is each benchmark's lift distinguishable from zero across seeds?
for name, acc, flo in (("rjudge", "acc_rjudge", "floor_rjudge"),
                       ("asb", "acc_asb", "floor_asb")):
    lifts = [data[s][acc]-data[s][flo] for s in seeds]
    m = st.mean(lifts); sd = st.stdev(lifts) if len(lifts) > 1 else 0.0
    se = sd/ (len(lifts)**0.5) if len(lifts) > 1 else float('inf')
    t = m/se if se > 0 else float('inf')
    print(f"  {name:7s} mean lift {m*100:+.1f}pt  sd {sd*100:.1f}  t={t:+.2f}"
          f"  -> {'CLEARS' if m>0 and abs(t)>2 else 'not established'}")
PY
echo "NSEED_DONE"
