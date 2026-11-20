#!/usr/bin/env bash
# Teacher-written rewrites of a line range of the Apollo corpus (20 Nov 2026).
# Two GPUs, vLLM 0.17, same environment as precompute_logits.slurm.
#   sbatch --export=ALL,START=0,COUNT=50000,OUT=data/raw/rewrites/part00.jsonl scripts/generate_rewrites.slurm
#SBATCH --job-name=rewrites
#SBATCH --account=brics.u6wc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6wc/savants/logs/rewrites_%j.log

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
export CUDAHOSTCXX="$SAVANTS/gcc13/bin/aarch64-conda-linux-gnu-g++"
export CXX="$CUDAHOSTCXX"
export CC="$SAVANTS/gcc13/bin/aarch64-conda-linux-gnu-gcc"
unset NVCC_PREPEND_FLAGS
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# shellcheck disable=SC1091
source "$SAVANTS/venv-vllm126/bin/activate"
NVIDIA_LIBS=$(find "$SAVANTS/venv-vllm126/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$SAVANTS/gcc13/lib:${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "visible gpus: $NGPU"
if [[ "$NGPU" -lt 2 ]]; then echo "need 2 GPUs, saw $NGPU"; exit 1; fi

: "${START:?}" "${COUNT:?}" "${OUT:?}"
mkdir -p "$(dirname "$SAVANTS/$OUT")"
ARGS=()
if [[ -n "${STYLES:-}" ]]; then ARGS+=(--styles "${STYLES//:/,}"); fi
if [[ -n "${MAX_NEW:-}" ]]; then ARGS+=(--max-new "$MAX_NEW"); fi
# The script resumes from its output file, so a crashed engine (vLLM worker
# hangs, 20 Nov 2026) just costs the batch in flight: retry a few times.
for attempt in 1 2 3 4 5; do
    echo "attempt $attempt $(date -u +%FT%TZ)"
    if python "$SAVANTS/scripts/generate_rewrites.py" \
        --corpus "$SAVANTS/data/raw/apollo/merged_apollo_en_corpus.jsonl" \
        --model "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
        --out "$SAVANTS/$OUT" --start "$START" --count "$COUNT" --seed "$((START + attempt))" "${ARGS[@]}"; then
        exit 0
    fi
    echo "attempt $attempt failed"; sleep 30
done
exit 1
