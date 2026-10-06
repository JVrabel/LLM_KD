#!/usr/bin/env bash
# Packs the clean Python JSONL into 4096-token uint32 shards. CPU only.
#SBATCH --job-name=pack_python
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6wc/savants/logs/pack_python_%j.log

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
if ! python -c "import transformers" 2>/dev/null; then
    uv pip install "transformers>=4.51"
fi
python "$SAVANTS/scripts/pack_shards.py" \
    --jsonl "$SAVANTS/data/raw/python/python_clean.jsonl" \
    --tokenizer "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "$SAVANTS/data/packed/python"
