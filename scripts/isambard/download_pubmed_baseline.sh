#!/usr/bin/env bash
# Downloads the NLM PubMed annual baseline (XML) and verifies every file against its
# published MD5. Safe to re-run: verified files are marked with <name>.ok and skipped.
# Usage: download_pubmed_baseline.sh OUT_DIR   (env PARALLEL, default 3; NCBI throttles
# heavier parallelism and then serves truncated files)
set -euo pipefail

BASE_URL="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
OUT_DIR="${1:?usage: $0 OUT_DIR}"
PARALLEL="${PARALLEL:-3}"
umask 002

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

download_one() {
    local name="$1"
    [[ -f "$name.ok" ]] && return 0

    local expected actual attempt
    for attempt in 1 2 3; do
        curl -fsS --retry 5 --retry-delay 10 -o "$name.md5" "$BASE_URL/$name.md5" || continue
        curl -fsS --retry 5 --retry-delay 10 -o "$name.part" "$BASE_URL/$name" || continue
        expected=$(sed -E 's/.*= *([0-9a-f]{32}).*/\1/' "$name.md5")
        actual=$(md5sum "$name.part" | cut -d' ' -f1)
        if [[ "$expected" == "$actual" ]]; then
            mv "$name.part" "$name"
            touch "$name.ok"
            return 0
        fi
        echo "MD5 mismatch (attempt $attempt): $name" >&2
        sleep 10
    done
    rm -f "$name.part"
    echo "FAILED: $name" >&2
    return 1
}
export -f download_one
export BASE_URL

curl -fsS "$BASE_URL/" | grep -oE 'pubmed[0-9]+n[0-9]+\.xml\.gz' | sort -u > files.txt
echo "[$(date -Is)] $(wc -l < files.txt) files listed"

xargs -P "$PARALLEL" -I{} bash -c 'download_one "$1"' _ {} < files.txt || true

total=$(wc -l < files.txt)
verified=$(ls -1 ./*.xml.gz.ok 2>/dev/null | wc -l)
echo "[$(date -Is)] verified $verified / $total"
[[ "$verified" -eq "$total" ]]
