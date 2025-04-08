#!/bin/bash
#SBATCH --job-name=LLM_test_example
#SBATCH --output=/scratch/%u/logs/training_output_%j.log
#SBATCH --error=/scratch/%u/logs/training_error_%j.log
#SBATCH --gres=gpu:4           # Request 4 GPUs instead of 1
#SBATCH --mem=400G            # Increased memory for 4 GPUs
#SBATCH --time=32:00:00
#SBATCH --ntasks=1

# Enable debugging
set -x

# Start container using /scratch
enroot start --root --mount=/scratch/vrabel/LLM_KD:/workspace pytorch_LLM_v1 bash -c "
    echo 'Inside container: Mounted directory contents:' &&
    ls -al /workspace &&
    mkdir -p /workspace/logs &&
    python src/train_distr.py --config src/config.yaml
"
