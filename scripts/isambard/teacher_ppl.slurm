#!/usr/bin/env bash
# Installs vLLM (CUDA, aarch64) if needed and scores the Phase 1 teacher.
#SBATCH --job-name=teacher_ppl
#SBATCH --account=brics.u6wc
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6wc/savants/logs/teacher_ppl_%j.log

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

cd "$SAVANTS"
# The GPU driver (565.57) supports CUDA 12.6, not the CUDA 12.9/13 build that
# current vLLM pulls by default. vLLM 0.17 is the first release with Qwen3.5
# and it pins torch 2.10, which has a cu126 aarch64 wheel.
VENV=venv-vllm126
if [[ ! -x $VENV/bin/python ]]; then
    uv venv "$VENV" --python 3.12
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
if ! python -c "import vllm, torch; assert torch.version.cuda.startswith('12.6')" 2>/dev/null; then
    uv pip install "vllm==0.17.0"
    uv pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
        --index-url https://download.pytorch.org/whl/cu126 --reinstall
fi
NVIDIA_LIBS=$(find "$VENV/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "host compiler: $($CXX --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python "$SAVANTS/scripts/teacher_ppl.py"
