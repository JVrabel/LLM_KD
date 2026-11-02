#!/usr/bin/env bash
# Benchmark teacher prompt-logprob throughput without writing output shards.
#SBATCH --job-name=logit_bench
#SBATCH --account=brics.u6wc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6wc/savants/logs/logit_bench_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
SAVANTS="$PROJECTDIR/savants"
export CUDAHOSTCXX="$SAVANTS/gcc13/bin/aarch64-conda-linux-gnu-g++"
export CXX="$CUDAHOSTCXX"
export CC="$SAVANTS/gcc13/bin/aarch64-conda-linux-gnu-gcc"
unset NVCC_PREPEND_FLAGS
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# shellcheck disable=SC1091
source "$SAVANTS/venv-vllm126/bin/activate"
NVIDIA_LIBS=$(find "$SAVANTS/venv-vllm126/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$SAVANTS/gcc13/lib:${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "nodes: ${SLURM_JOB_NODELIST:-local} visible_gpus: $NGPU cuda_visible: ${CUDA_VISIBLE_DEVICES:-unset}"
if [[ "$NGPU" -lt 2 ]]; then
    echo "need 2 GPUs on one node, saw $NGPU"
    exit 1
fi

python "$SAVANTS/scripts/benchmark_logits.py" \
    --shards "$SAVANTS/data/packed/python" \
    --model "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --batch-sizes 4 8 16 \
    --sequences 16
