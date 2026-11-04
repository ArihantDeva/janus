#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=janusgguf
#SBATCH --exclude=gpuk001,gpuk002,gpuk003,gpuk004,gpuk005,gpuk006,gpuk007,gpuk008,gpuk009,gpuk010,gpuk011,gpuk012,gpuk013,gpuk014,gpuk015,gpuk016,gpuk017,gpuk018
#SBATCH --output=/scratch/ad2416/janusgguf.out
# Produce the Q4_K_M GGUF that llama-server serves and that the goal names.
#
# Q4_K_M only, never Q2: below 4 bits the quantisation noise swamps the very
# probability difference this whole approach measures. A cheaper file would be
# cheaper and would quietly destroy the signal, so it is not offered.
# The build links against HPC SDK CUDA 12.3; those libs are not on the default
# loader path, so llama-quantize starts and dies on "libcudart.so.12: cannot open
# shared object file". Export it once here rather than rediscovering it per tool.
export LD_LIBRARY_PATH=/opt/sw/packages/hpc_sdk/Linux_x86_64/2023/cuda/lib64:/opt/sw/packages/hpc_sdk/Linux_x86_64/2023/math_libs/12.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_HOME=/scratch/ad2416/hf
CONV=/scratch/ad2416/llamasrc/llama.cpp-master/convert_hf_to_gguf.py
QUANT=/scratch/ad2416/llamasrc/llama.cpp-master/build/bin/llama-quantize
OUT=/scratch/ad2416/models
mkdir -p "$OUT" || exit 1

MERGED=${MERGED:-/scratch/ad2416/llama-1b-merged}
if [ ! -f "$MERGED/config.json" ]; then
  /scratch/ad2416/venv-torch/bin/python -u /scratch/ad2416/janussrc/merge_adapter.py \
    --adapter /scratch/ad2416/janusout-fix/adapter_seed0 \
    --out "$MERGED" || exit 2
fi

/scratch/ad2416/venv-torch/bin/python -u "$CONV" "$MERGED" \
  --outfile "$OUT/llama-1b-guard-f16.gguf" --outtype f16 || exit 3

"$QUANT" "$OUT/llama-1b-guard-f16.gguf" \
  "$OUT/llama-1b-guard-q4_k_m.gguf" Q4_K_M 8 || exit 4

ls -la "$OUT"/*.gguf
echo GGUF_OK
