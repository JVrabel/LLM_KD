#!/usr/bin/env bash
# Submit a continuation run as chained 24 h training jobs (RESUME_LATEST=1,
# afterany), then answer-text exams on the listed checkpoints and held-out
# loss sweeps, all afterok on the last training job. Generalises submit_3b.sh
# (9 Nov 2026). Run on the cluster from /projects/u6wc/savants:
#   NAME=apollo_ntp_1.7b_3b CONFIG=config/pilot_1.7b_medical_apollo_ntp_shuffled_3b.yaml \
#   RUN_DIR=runs/pilot_1.7b_medical_apollo_ntp_shuffled_3b JOBS=4 \
#   EVAL_STEPS="60000 91551" SWEEPS="data/packed/medical_apollo:0 4 8 12 16 20 24 28 31" \
#   bash scripts/submit_chain.sh
# SWEEPS is a list of PACKED:SHARDS entries separated by ';'. AFTER=<jobid>
# gates the first training job with afterok (smoke test). Sweep output:
# logs/diag/forget_<run name>[_<pack name>].json; exam output:
# logs/<NAME>_step<step>_mmlu.json.
set -euo pipefail
: "${NAME:?}" "${CONFIG:?}" "${RUN_DIR:?}" "${JOBS:?}" "${EVAL_STEPS:?}"
S=/projects/u6wc/savants
cd "$S"
CONFIG="$S/${CONFIG#"$S"/}"
RUN_DIR="$S/${RUN_DIR#"$S"/}"
RUN_NAME=$(basename "$RUN_DIR")

prev="${AFTER:-}"
for i in $(seq 1 "$JOBS"); do
    dep=()
    if [[ "$i" == 1 && -n "$prev" ]]; then dep=(--dependency="afterok:$prev"); elif [[ -n "$prev" ]]; then dep=(--dependency="afterany:$prev"); fi
    jid=$(sbatch --parsable --job-name="$NAME" "${dep[@]}" \
        --output="$S/logs/${NAME}_%j.log" \
        --export=ALL,CONFIG="$CONFIG",RESUME_LATEST=1 \
        scripts/train_pilot.slurm)
    echo "train job $i: $jid"
    prev="$jid"
done

for step in $EVAL_STEPS; do
    ckpt="$RUN_DIR/step_$(printf '%07d' "$step").pt"
    jid=$(sbatch --parsable --job-name="mmlu_${NAME}_${step}" --dependency="afterok:$prev" \
        --output="$S/logs/mmlu_${NAME}_${step}_%j.log" \
        --export=ALL,CHECKPOINT="$ckpt",OUT="$S/logs/${NAME}_step${step}_mmlu.json" \
        scripts/student_mmlu.slurm)
    echo "mmlu step $step: $jid"
done

IFS=';' read -ra sweeps <<< "${SWEEPS:-}"
for entry in "${sweeps[@]}"; do
    packed="${entry%%:*}"
    shards="${entry#*:}"
    packed="$S/${packed#"$S"/}"
    pack_name=$(basename "$packed")
    out="$S/logs/diag/forget_${RUN_NAME}_${pack_name}.json"
    jid=$(sbatch --parsable --job-name="forget_${NAME}_${pack_name}" --dependency="afterok:$prev" --time=02:00:00 \
        --output="$S/logs/forget_${NAME}_${pack_name}_%j.log" \
        --export=ALL,RUN_DIR="$RUN_DIR",PACKED="$packed",STEPS="$EVAL_STEPS",SHARDS="$shards",OUT="$out" \
        scripts/diag_forgetting.slurm)
    echo "held-out sweep on $pack_name: $jid"
done
