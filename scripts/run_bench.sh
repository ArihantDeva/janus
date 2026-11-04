#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=janusbench
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=/scratch/ad2416/janusbench.out
# Score the three 1B adapters on R-Judge and Agent-SafetyBench. This is the
# benchmark criterion of the goal, and it needs no retraining: the adapters
# already exist. Runs on its own GPU while the Glimmer QLoRA job holds another.
export JANUS_DATA=/scratch/ad2416/janusdata
export HF_HOME=/scratch/ad2416/hf
cd /scratch/ad2416
# A loop that ignores its children's exit codes reports success no matter how
# badly they failed. The first bench job exited 0 with COMPLETED while every one
# of the three seeds had crashed on a NameError, and the only reason it was
# caught is that 49 seconds is implausible for 3 seeds x 2 benchmarks. An
# implausible runtime was the alarm; the exit code should have been.
rc=0
for s in 0 1 2; do
  echo "########## seed $s ##########"
  /scratch/ad2416/venv-torch/bin/python -u /scratch/ad2416/janussrc/eval_bench.py \
    --adapter /scratch/ad2416/janusout-fix/adapter_seed$s \
    --bench both \
    --out /scratch/ad2416/janusout-fix/bench_seed$s.json
  this=$?
  if [ "$this" -ne 0 ]; then
    echo "SEED $s FAILED rc=$this"
    rc=$this
  fi
done
if [ "$rc" -ne 0 ]; then
  echo "BENCH_FAILED rc=$rc"
  exit "$rc"
fi
echo BENCH_DONE
