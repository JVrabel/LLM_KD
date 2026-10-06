#!/usr/bin/env bash
# Single-GPU 0.6B student. CONFIG is the yaml, copied under savants/config.
#SBATCH --job-name=pilot_0.6b
#SBATCH --account=brics.u6wc
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6wc/savants/logs/pilot_0.6b_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.6
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1
CONFIG="${CONFIG:-$SAVANTS/config/pilot_0.6b_smoke.yaml}"

cd "$SAVANTS"
if [[ ! -x venv-train/bin/python ]]; then
    uv venv venv-train --python 3.12
fi
# shellcheck disable=SC1091
source venv-train/bin/activate
if ! python -c "import torch, transformers; assert torch.version.cuda.startswith('12.6')" 2>/dev/null; then
    uv pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu126
    uv pip install "transformers>=4.51" pyyaml
fi
NVIDIA_LIBS=$(find venv-train/lib/python3.12/site-packages/nvidia -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python "$SAVANTS/src/train_kd.py" --config "$CONFIG"
