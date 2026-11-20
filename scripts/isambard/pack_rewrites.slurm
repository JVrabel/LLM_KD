#!/usr/bin/env bash
# Pack the teacher-written rewrites (data/raw/rewrites/part*.jsonl) into
# shards in a seeded random order (20 Nov 2026). MAX_SHARDS caps the pack.
#SBATCH --job-name=pack_rewrites
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/u6wc/savants/logs/pack_rewrites_%j.log
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1
cd "$SAVANTS"
# shellcheck disable=SC1091
source venv-data/bin/activate
ARGS=()
if [[ -n "${MAX_SHARDS:-}" ]]; then ARGS+=(--max-shards "$MAX_SHARDS"); fi
python "$SAVANTS/scripts/pack_shards.py" \
    --jsonl "$SAVANTS"/data/raw/rewrites/part*.jsonl \
    --tokenizer "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "${OUT:-$SAVANTS/data/packed/rewrites}" \
    --shuffle-seed 11 "${ARGS[@]}"
