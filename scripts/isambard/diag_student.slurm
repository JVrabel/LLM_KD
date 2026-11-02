#!/usr/bin/env bash
# Held-out loss, teacher KL, exam calibration and cloze diagnostics for one checkpoint. One GPU.
#SBATCH --job-name=diag_student
#SBATCH --account=brics.u6wc
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6wc/savants/logs/diag_student_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.6
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$SAVANTS"
JOB_CACHE="$SAVANTS/job-cache/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_CACHE/tmp" "$JOB_CACHE/torchinductor" "$JOB_CACHE/triton"
export TMPDIR="$JOB_CACHE/tmp"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE/triton"
# shellcheck disable=SC1091
source venv-train/bin/activate
NVIDIA_LIBS=$(find venv-train/lib/python3.12/site-packages/nvidia -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUT:?set OUT}"
python "$SAVANTS/scripts/diag_student.py" --checkpoint "$CHECKPOINT" --out "$OUT" ${DIAG_ARGS:-}
