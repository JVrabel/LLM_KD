#!/usr/bin/env bash
# Extracts PubMed abstracts, drops duplicates and eval contamination, and packs
# the medical mix (Apollo + PubMed) into the same shard format as the Python set.
#SBATCH --job-name=build_medical
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=/projects/u6wc/savants/logs/build_medical_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1

cd "$SAVANTS"
if [[ ! -x venv-data/bin/python ]]; then
    uv venv venv-data --python 3.12
fi
# shellcheck disable=SC1091
source venv-data/bin/activate
if ! python -c "import transformers, datasets" 2>/dev/null; then
    uv pip install "transformers>=4.51" "datasets>=3"
fi

python "$SAVANTS/scripts/extract_pubmed.py" \
    --xml-dir "$SAVANTS/data/raw/pubmed_baseline" \
    --out-dir "$SAVANTS/data/raw/medical/pubmed"
python "$SAVANTS/scripts/filter_medical.py" \
    --pubmed-dir "$SAVANTS/data/raw/medical/pubmed" \
    --apollo "$SAVANTS/data/raw/apollo/merged_apollo_en_corpus.jsonl" \
    --eval-dir "$SAVANTS/data/raw/eval" \
    --out "$SAVANTS/data/raw/medical/medical_clean.jsonl"
python "$SAVANTS/scripts/pack_shards.py" \
    --jsonl "$SAVANTS/data/raw/medical/medical_clean.jsonl" \
    --tokenizer "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "$SAVANTS/data/packed/medical"
