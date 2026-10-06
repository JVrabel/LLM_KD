#!/usr/bin/env bash
# Downloads the Phase 1 teacher, the clinical guidelines corpus and the evaluation
# sets from Hugging Face into the shared project directory on Isambard-AI.
# Run on the login node (compute nodes have no internet). Safe to re-run: hf download
# skips files that are already complete.
set -euo pipefail

SAVANTS_DIR="${SAVANTS_DIR:-$PROJECTDIR/savants}"
export HF_HOME="$SAVANTS_DIR/hf_cache"
export HF_XET_HIGH_PERFORMANCE=1
export PATH="$HOME/.local/bin:$PATH"
umask 002

hf() {
    uvx --python 3.12 --from "huggingface_hub[hf_xet]>=1.0" hf "$@"
}

EVAL_DATASETS=(
    GBaker/MedQA-USMLE-4-options
    openlifescienceai/medmcqa
    qiaojin/PubMedQA
    cais/mmlu
    allenai/ai2_arc
    baber/piqa
    Rowan/hellaswag
)

echo "[$(date -Is)] Qwen3.5-35B-A3B-Base"
hf download Qwen/Qwen3.5-35B-A3B-Base \
    --local-dir "$SAVANTS_DIR/models/Qwen3.5-35B-A3B-Base"

echo "[$(date -Is)] epfl-llm/guidelines"
hf download epfl-llm/guidelines --repo-type dataset \
    --local-dir "$SAVANTS_DIR/data/raw/guidelines"

for repo in "${EVAL_DATASETS[@]}"; do
    echo "[$(date -Is)] $repo"
    hf download "$repo" --repo-type dataset \
        --local-dir "$SAVANTS_DIR/data/raw/eval/${repo//\//__}"
done

echo "[$(date -Is)] DONE"
