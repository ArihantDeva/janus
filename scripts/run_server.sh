#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=janussrv
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=/scratch/ad2416/janussrv.out
# Serve the 1B guard over the OpenAI-compatible API that Muse Code speaks
# (--provider local --base-url). This is the transport the integration is
# measured against, so it must be the real llama.cpp server, not a shim.
#
# The build links against the HPC SDK CUDA 12.3 libraries, which are not on the
# default loader path -- without LD_LIBRARY_PATH the binary starts and then dies
# on "libcudart.so.12 => not found".
export LD_LIBRARY_PATH=/opt/sw/packages/hpc_sdk/Linux_x86_64/2023/cuda/lib64:/opt/sw/packages/hpc_sdk/Linux_x86_64/2023/math_libs/12.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

BIN=/scratch/ad2416/llamasrc/llama.cpp-master/build/bin
MODEL=${MODEL:-/scratch/ad2416/models/llama-1b-guard-q4_k_m.gguf}

exec "$BIN/llama-server" \
  --model "$MODEL" \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 2048 \
  --n-gpu-layers 99 \
  --parallel 4 \
  --log-disable
