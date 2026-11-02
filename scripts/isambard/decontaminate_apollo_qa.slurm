#!/usr/bin/env bash
# Filter exam test questions out of the Apollo question-answer file, then pack
# the clean file into data/packed/medical_apollo_qa_clean with the same shuffle
# seed as the original QA pack. CPU only, minutes. Teacher logits for the new
# pack are a separate precompute_logits job.
#SBATCH --job-name=decontaminate_qa
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6wc/savants/logs/decontaminate_qa_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1
QA="$SAVANTS/data/raw/apollo/qa"

cd "$SAVANTS"
# shellcheck disable=SC1091
source venv-data/bin/activate
python "$SAVANTS/scripts/decontaminate_qa.py" \
    --jsonl "$QA/merged_apollo_en_qa.jsonl" \
    --out "$QA/merged_apollo_en_qa_clean.jsonl" \
    --report "$QA/decontamination.json" \
    --medqa-test "$SAVANTS/data/raw/eval/GBaker__MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl" \
    --mmlu-dir "$SAVANTS/data/raw/eval/mmlu_medical"
python "$SAVANTS/scripts/pack_shards.py" \
    --jsonl "$QA/merged_apollo_en_qa_clean.jsonl" \
    --tokenizer "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "$SAVANTS/data/packed/medical_apollo_qa_clean" \
    --shuffle-seed 42
cat "$SAVANTS/data/packed/medical_apollo_qa_clean/meta.json"
