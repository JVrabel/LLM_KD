#!/usr/bin/env bash
# Apollo exam-style question-answer pairs (417k rows of [prompt, answer]), shuffled
# and packed into the same 4096-token shards as the other sets. CPU only, minutes.
#SBATCH --job-name=pack_apollo_qa
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6wc/savants/logs/pack_apollo_qa_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1

cd "$SAVANTS"
# shellcheck disable=SC1091
source venv-data/bin/activate
python "$SAVANTS/scripts/pack_shards.py" \
    --jsonl "$SAVANTS/data/raw/apollo/qa/merged_apollo_en_qa.jsonl" \
    --tokenizer "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "$SAVANTS/data/packed/medical_apollo_qa" \
    --shuffle-seed 42
