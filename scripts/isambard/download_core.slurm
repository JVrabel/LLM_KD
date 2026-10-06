#!/usr/bin/env bash
# Runs the core downloads on a compute node: login nodes kill background processes on
# logout (KillUserProcesses=yes), compute nodes have internet. Both scripts resume.
# Submit from $PROJECTDIR/savants: sbatch scripts/download_core.slurm
#SBATCH --job-name=savants_download
#SBATCH --account=brics.u6wc
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_core_%j.log

set -euo pipefail

SAVANTS_DIR="${SAVANTS_DIR:-$PROJECTDIR/savants}"
cd "$SAVANTS_DIR"

scripts/download_hf_core.sh > logs/download_hf_core.log 2>&1 &
hf_pid=$!
PARALLEL=3 scripts/download_pubmed_baseline.sh "$SAVANTS_DIR/data/raw/pubmed_baseline" \
    > logs/download_pubmed_baseline.log 2>&1 &
pubmed_pid=$!

status=0
wait "$hf_pid" || { echo "HF download failed"; status=1; }
wait "$pubmed_pid" || { echo "PubMed download failed"; status=1; }
echo "[$(date -Is)] finished with status $status"
exit "$status"
