#!/usr/bin/env bash
# Submit the 3B-token continuation of the shuffled distilled students (4 Nov
# 2026): chained 24 h training jobs with RESUME_LATEST=1, then exams and the
# shard-31 held-out sweep on the checkpoints at about 1.5, 2.0, 2.5 and 3.0B
# tokens. Run on the cluster from /projects/u6wc/savants:
#   bash scripts/submit_3b.sh 0.6b 2      # two 24 h jobs (about 39 h)
#   bash scripts/submit_3b.sh 1.7b 3      # three 24 h jobs (about 68 h)
set -euo pipefail
SIZE="${1:?size, e.g. 0.6b}"
JOBS="${2:?number of chained 24 h training jobs}"
S=/projects/u6wc/savants
NAME="apollo_kd_${SIZE}_3b"
CONFIG="$S/config/pilot_${SIZE}_medical_apollo_kd_shuffled_3b.yaml"
RUN_DIR="$S/runs/pilot_${SIZE}_medical_apollo_kd_shuffled_3b"
cd "$S"

prev=""
for i in $(seq 1 "$JOBS"); do
    dep=()
    if [[ -n "$prev" ]]; then dep=(--dependency="afterany:$prev"); fi
    jid=$(sbatch --parsable --job-name="$NAME" "${dep[@]}" \
        --output="$S/logs/${NAME}_%j.log" \
        --export=ALL,CONFIG="$CONFIG",RESUME_LATEST=1 \
        scripts/train_pilot.slurm)
    echo "train job $i: $jid"
    prev="$jid"
done

for step in 45000 60000 75000 91551; do
    ckpt="$RUN_DIR/step_$(printf '%07d' "$step").pt"
    jid=$(sbatch --parsable --job-name="mmlu_${SIZE}_3b_${step}" --dependency="afterok:$prev" \
        --output="$S/logs/mmlu_${SIZE}_3b_${step}_%j.log" \
        --export=ALL,CHECKPOINT="$ckpt",OUT="$S/logs/apollo_kd_${SIZE}_3b_step${step}_mmlu.json" \
        scripts/student_mmlu.slurm)
    echo "mmlu step $step: $jid"
done

jid=$(sbatch --parsable --job-name="forget_${SIZE}_3b" --dependency="afterok:$prev" --time=02:00:00 \
    --output="$S/logs/forget_${SIZE}_3b_%j.log" \
    --export=ALL,RUN_DIR="$RUN_DIR",PACKED="$S/data/packed/medical_apollo",STEPS="45000 60000 75000 91551",SHARDS="0 4 8 12 16 20 24 28 31",OUT="$S/logs/diag/forget_pilot_${SIZE}_medical_apollo_kd_shuffled_3b.json" \
    scripts/diag_forgetting.slurm)
echo "forget sweep: $jid"
