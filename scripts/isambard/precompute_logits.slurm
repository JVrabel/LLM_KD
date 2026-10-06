#!/usr/bin/env bash
# Top-64 teacher logprobs for packed shards. Two GPUs, vLLM 0.17.
# MAX_SHARDS=1 scores only the first shard (the smoke-sized KD run).
#SBATCH --job-name=teacher_logits
#SBATCH --account=brics.u6wc
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6wc/savants/logs/teacher_logits_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
module load gcc-native/13.2
module load cuda/12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDAHOSTCXX="$(command -v g++)"
export CXX="$CUDAHOSTCXX"
export CC="$(command -v gcc)"
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# shellcheck disable=SC1091
source "$SAVANTS/venv-vllm126/bin/activate"
NVIDIA_LIBS=$(find "$SAVANTS/venv-vllm126/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ARGS=()
if [[ -n "${MAX_SHARDS:-}" ]]; then ARGS+=(--max-shards "$MAX_SHARDS"); fi
python "$SAVANTS/scripts/precompute_logits.py" \
    --shards "${SHARDS:-$SAVANTS/data/packed/python}" \
    --model "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "${LOGIT_DIR:-$SAVANTS/data/logits/python}" \
    "${ARGS[@]}"
