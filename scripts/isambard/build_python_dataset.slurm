#!/usr/bin/env bash
# Builds the clean Python dataset on a compute node (login nodes kill processes on
# logout, and this runs for many hours). Resumable: re-running continues from the
# files already written.
# Submit from anywhere: sbatch scripts/build_python_dataset.slurm
#SBATCH --job-name=python_dataset
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6wc/savants/logs/python_dataset_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SAVANTS_DIR="${SAVANTS_DIR:-$PROJECTDIR/savants}"

EXTRA=()
if [[ -n "${MAX_FILES:-}" ]]; then EXTRA+=(--max-files "$MAX_FILES"); fi
if [[ -n "${TOKEN_BUDGET:-}" ]]; then EXTRA+=(--token-budget "$TOKEN_BUDGET"); fi
if [[ -n "${MIN_SCORE:-}" ]]; then EXTRA+=(--min-score "$MIN_SCORE"); fi

uv run --python 3.12 --with boto3 --with datasets --with transformers \
    "$SAVANTS_DIR/scripts/build_python_dataset.py" \
    --out "${OUT_DIR:-$SAVANTS_DIR/data/raw/python}" \
    --tokenizer "$SAVANTS_DIR/models/Qwen3.5-35B-A3B-Base" \
    "${EXTRA[@]}"
