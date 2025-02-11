#!/bin/bash
#SBATCH --job-name=LLM_test_example
#SBATCH --output=/scratch/%u/logs/training_output_%j.log
#SBATCH --error=/scratch/%u/logs/training_error_%j.log
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4

# Enable debugging
set -x

# Set environment variables for distributed training
master_port=$(shuf -i 29500-29999 -n 1)
master_addr=$(hostname)

# Start container using /scratch
enroot start --root --mount=/scratch/vrabel/LLM_KD:/workspace pytorch_LLM_v1 bash -c "
    echo 'Inside container: Mounted directory contents:' &&
    ls -al /workspace &&
    mkdir -p /workspace/logs &&
    PYTHONFAULTHANDLER=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    NCCL_DEBUG=INFO \
    TORCHELASTIC_ERROR_FILE=/workspace/logs/elastic_error.log \
    MASTER_PORT=${master_port} \
    MASTER_ADDR=${master_addr} \
    WORLD_SIZE=4 \
    OMP_NUM_THREADS=4 \
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=${master_port} \
        --master_addr=${master_addr} \
        --node_rank=0 \
        --nnodes=1 \
        src/train_distr.py \
        --config src/config.yaml 2>&1 | tee /workspace/logs/training_detailed.log
"
