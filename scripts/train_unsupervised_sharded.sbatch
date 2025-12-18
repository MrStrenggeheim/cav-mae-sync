#!/bin/bash
#SBATCH --job-name="Train CAV-MAE-Sync Sharded"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/hunecke/fakesync/slurm_logs/slurm-cav-mae-sync-train-sharded-%j.out
#SBATCH --error=/storage/slurm/hunecke/fakesync/slurm_logs/slurm-cav-mae-sync-train-sharded-%j.err

# 1. Force stdout/stderr to flush immediately
export PYTHONUNBUFFERED=1
# DEBUG for profiling output
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "================ DEBUG INFO ================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# 2. Environment Setup
source /storage/slurm/hunecke/fakesync/.venv/bin/activate
echo "Running python version $(python3 --version) from $(which python)"
echo "GPU info:"
nvidia-smi

# 3. Paths
PROJECT_PATH=/storage/slurm/hunecke/fakesync/cav-mae-sync
SHARDED_DATASET_DIR=/storage/slurm/hunecke/fakesync/data/voxceleb2/preprocessed/shards_train
SAVE_PATH=${PROJECT_PATH}/outputs/checkpoints/unsupervised_train_sharded_$(date +%Y%m%d_%H%M%S)

# 4. Training Configuration
BATCH_SIZE=20
TOTAL_FRAME=16
EPOCHS=10
WARMUP_EPOCHS=1
LR=1e-4
# Gradient accumulation: effective_batch = BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS
# Default: 24 * 2 * 2 = 96 effective batch
ACCUMULATION_STEPS=2

# 5. Multi-GPU / NCCL Optimizations
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_NCCL_BLOCKING_WAIT=1
# Use unique port per job to avoid "Address already in use" errors
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "Starting training..."
echo "Project Path: ${PROJECT_PATH}"
echo "Sharded Dataset Dir: ${SHARDED_DATASET_DIR}"
echo "Save Path: ${SAVE_PATH}"

python3 ${PROJECT_PATH}/train_unsupervised.py \
    --sharded_dataset_dir ${SHARDED_DATASET_DIR} \
    --batch_size ${BATCH_SIZE} \
    --total_frame ${TOTAL_FRAME} \
    --target_length 48 \
    --audio_length 48 \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --lr ${LR} \
    --save_path ${SAVE_PATH} \
    --resume /storage/slurm/schnackl/fakesync/cav-mae-sync/pretrained_models/cav_mae_sync.pth \
    --contrast_bidirect \
    --contrast_loss_weight 0.5 \
    --contrast_inter_weight 0.3 \
    --contrast_intra_weight 0.7 \
    --mae_loss_weight 1.0 \
    --log_freq 100 \
    --cls_token \
    --num_register_tokens 8 \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS}
    # --fast_dev_run \

echo "Training finished at $(date) with exit code $?"
