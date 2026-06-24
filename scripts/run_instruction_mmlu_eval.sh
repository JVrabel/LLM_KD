#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

CHECKPOINT="${CHECKPOINT:-outputs/instruction_tuning/instruction_run_20260610_083645/checkpoints/best_model.pt}"
CONFIG="${CONFIG:-config/instruction_mmlu.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/instruction_tuned_medical}"
TEMP_DIR="${TEMP_DIR:-}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
if [[ -n "$TEMP_DIR" ]]; then
  mkdir -p "$TEMP_DIR"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/eval_instruction_tuned_medical_${timestamp}.log"

echo "Checkpoint: $CHECKPOINT"
echo "Config:     $CONFIG"
echo "Output:     $OUTPUT_DIR"
echo "Temp dir:   ${TEMP_DIR:-default}"
echo "GPU:        $GPU"
echo "Log:        $log_file"

cmd=(
  .venv/bin/python src/eval.py
  --checkpoint "$CHECKPOINT"
  --config "$CONFIG"
  --output_dir "$OUTPUT_DIR"
  --medical_only
)

if [[ -n "$TEMP_DIR" ]]; then
  cmd+=(--temp_dir "$TEMP_DIR")
fi

CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee "$log_file"
