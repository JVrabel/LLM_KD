#!/usr/bin/env bash
# Teacher top-64 logits for a list of shards of one pack, as CHAINS parallel
# chains of JOBS chained 24 h two-GPU jobs each (9 Nov 2026). The shard list is
# dealt round-robin to the chains; a chain's later jobs skip shards that are
# already done, so an over-provisioned chain just ends early. Run from
# /projects/u6wc/savants:
#   PACK=data/packed/medical LOGIT_DIR=data/logits/medical \
#   SHARDS="33,36,...,286" CHAINS=4 JOBS=4 bash scripts/submit_logits.sh
# One shard (8192 x 4096 tokens) takes about 3.1 h at 3.0k tok/s, so a 24 h job
# scores about 7 shards. The list reaches the job colon-separated: sbatch
# --export splits its value on commas (10 Nov 2026: the first submission scored
# one shard per chain because of that).
set -euo pipefail
: "${PACK:?}" "${LOGIT_DIR:?}" "${SHARDS:?}" "${CHAINS:?}" "${JOBS:?}"
S=/projects/u6wc/savants
cd "$S"
PACK="$S/${PACK#"$S"/}"
LOGIT_DIR="$S/${LOGIT_DIR#"$S"/}"
NAME="${NAME:-logits_$(basename "$PACK")}"
mkdir -p "$LOGIT_DIR"
IFS=',' read -ra all <<< "$SHARDS"
for c in $(seq 0 $((CHAINS - 1))); do
    mine=()
    for i in "${!all[@]}"; do
        if (( i % CHAINS == c )); then mine+=("${all[$i]}"); fi
    done
    list=$(IFS=':'; echo "${mine[*]}")
    echo "chain $c shards: $list"
    prev=""
    for j in $(seq 1 "$JOBS"); do
        dep=()
        if [[ -n "$prev" ]]; then dep=(--dependency="afterany:$prev"); fi
        jid=$(sbatch --parsable --job-name="${NAME}_c${c}" "${dep[@]}" \
            --output="$S/logs/${NAME}_c${c}_%j.log" \
            --export=ALL,SHARDS="$PACK",LOGIT_DIR="$LOGIT_DIR",SHARD_NUMBERS="$list" \
            scripts/precompute_logits.slurm)
        echo "  job $j: $jid"
        prev="$jid"
    done
    echo "$prev" >> "$S/logs/${NAME}_last_jobs.txt"
done
