#!/usr/bin/env bash
# 0-shot and 5-shot letter scores for medical MMLU and MedQA. Two GPUs.
#SBATCH --job-name=teacher_mc
#SBATCH --account=brics.u6wc
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/projects/u6wc/savants/logs/teacher_mc_%j.log

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

python "$SAVANTS/scripts/teacher_mc_eval.py"
