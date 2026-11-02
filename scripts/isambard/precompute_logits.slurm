#!/usr/bin/env bash
# Top-64 teacher logprobs for packed shards. Two GPUs, vLLM 0.17.
# MAX_SHARDS=1 scores only the first shard (the smoke-sized KD run).
#SBATCH --job-name=teacher_logits
#SBATCH --account=brics.u6wc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6wc/savants/logs/teacher_logits_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
SAVANTS="$PROJECTDIR/savants"
JOB_CACHE="$SAVANTS/job-cache/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_CACHE/tmp" "$JOB_CACHE/torchinductor" "$JOB_CACHE/triton"
export TMPDIR="$JOB_CACHE/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE/triton"
# System GCC 13 was removed. This conda-forge 13.4 is the host compiler nvcc accepts.
export CUDAHOSTCXX="$SAVANTS/gcc13/bin/aarch64-conda-linux-gnu-g++"
export CXX="$CUDAHOSTCXX"
export CC="$SAVANTS/gcc13/bin/aarch64-conda-linux-gnu-gcc"
unset NVCC_PREPEND_FLAGS
GCC_VER="$("$CXX" -dumpfullversion 2>/dev/null || "$CXX" -dumpversion)"
echo "host compiler: $CXX $GCC_VER"
echo "nodes: ${SLURM_JOB_NODELIST:-local} gpus_on_node: ${SLURM_GPUS_ON_NODE:-unset} cuda_visible: ${CUDA_VISIBLE_DEVICES:-unset}"
case "$GCC_VER" in
    13*) ;;
    *) echo "flashinfer needs gcc 13, got $GCC_VER"; exit 1 ;;
esac
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# shellcheck disable=SC1091
source "$SAVANTS/venv-vllm126/bin/activate"
NVIDIA_LIBS=$(find "$SAVANTS/venv-vllm126/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$SAVANTS/gcc13/lib:${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "visible gpus: $NGPU"
if [[ "$NGPU" -lt 2 ]]; then
    echo "need 2 GPUs on one node, saw $NGPU"
    exit 1
fi

ARGS=()
if [[ -n "${MAX_SHARDS:-}" ]]; then ARGS+=(--max-shards "$MAX_SHARDS"); fi
python "$SAVANTS/scripts/precompute_logits.py" \
    --shards "${SHARDS:-$SAVANTS/data/packed/python}" \
    --model "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "${LOGIT_DIR:-$SAVANTS/data/logits/python}" \
    "${ARGS[@]}"
