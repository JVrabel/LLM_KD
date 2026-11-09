#!/usr/bin/env bash
# Cosmopedia v2 parquet -> jsonl -> 9 shards (0.3B tokens) in a seeded random
# document order, same shard format as the other packs (9 Nov 2026).
#SBATCH --job-name=pack_cosmo
#SBATCH --account=brics.u6wc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6wc/savants/logs/pack_cosmo_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SAVANTS="$PROJECTDIR/savants"
export UV_CACHE_DIR="$SAVANTS/uv-cache"
export HF_HUB_OFFLINE=1

cd "$SAVANTS"
# shellcheck disable=SC1091
source venv-data/bin/activate
python - <<'PY'
import glob, json, os
import pyarrow.parquet as pq
out = "data/raw/cosmopedia/cosmopedia_v2.jsonl"
if not os.path.exists(out + ".ok"):
    n = 0
    with open(out, "w") as handle:
        for path in sorted(glob.glob("data/raw/cosmopedia/cosmopedia-v2/*.parquet")):
            table = pq.read_table(path, columns=["text"])
            for text in table.column("text").to_pylist():
                if text and text.strip():
                    handle.write(json.dumps({"text": text}) + "\n")
                    n += 1
            print(path, n, flush=True)
    open(out + ".ok", "w").close()
    print("rows", n)
PY
python "$SAVANTS/scripts/pack_shards.py" \
    --jsonl "$SAVANTS/data/raw/cosmopedia/cosmopedia_v2.jsonl" \
    --tokenizer "$SAVANTS/models/Qwen3.5-35B-A3B-Base" \
    --out "$SAVANTS/data/packed/cosmopedia" \
    --shuffle-seed 7 --max-shards 9
