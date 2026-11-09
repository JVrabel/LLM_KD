#!/usr/bin/env bash
# Two of the 104 Cosmopedia v2 parquet files (about 0.54B tokens of synthetic
# textbooks, WikiHow articles and stories; HuggingFaceTB/smollm-corpus, ODC-BY).
# Login node, resumable. The pack job then takes a 0.3B-token random subset.
set -euo pipefail
SAVANTS_DIR="${SAVANTS_DIR:-$PROJECTDIR/savants}"
export HF_HOME="$SAVANTS_DIR/hf_cache"
export HF_XET_HIGH_PERFORMANCE=1
export PATH="$HOME/.local/bin:$PATH"
umask 002
uvx --python 3.12 --from "huggingface_hub[hf_xet]>=1.0" hf download HuggingFaceTB/smollm-corpus \
    --repo-type dataset \
    --include "cosmopedia-v2/train-00000-of-00104.parquet" \
    --include "cosmopedia-v2/train-00001-of-00104.parquet" \
    --local-dir "$SAVANTS_DIR/data/raw/cosmopedia"
ls -la "$SAVANTS_DIR/data/raw/cosmopedia/cosmopedia-v2"
