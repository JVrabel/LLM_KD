#!/usr/bin/env bash
# Apollo-only medical slice, then the same 4096-token packing as the other sets.
#SBATCH --job-name=pack_apollo
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6wc/savants/logs/pack_apollo_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1

cd "$SAVANTS"
# shellcheck disable=SC1091
source venv-data/bin/activate
python "$SAVANTS/scripts/filter_apollo.py" \
    --src "$SAVANTS/data/raw/medical/medical_clean.jsonl" \
    --out "$SAVANTS/data/raw/medical/apollo_only.jsonl"
python "$SAVANTS/scripts/pack_shards.py" \
    --jsonl "$SAVANTS/data/raw/medical/apollo_only.jsonl" \
    --tokenizer "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "$SAVANTS/data/packed/medical_apollo"
