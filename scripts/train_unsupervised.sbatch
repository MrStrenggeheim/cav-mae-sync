#!/bin/bash
#SBATCH --job-name="Train CAV-MAE-Sync Unsupervised"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1,VRAM:64G
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-train-%j.out
#SBATCH --error=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-train-%j.err

# 1. Force stdout/stderr to flush immediately
export PYTHONUNBUFFERED=1

echo "================ DEBUG INFO ================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# 2. Environment Setup
source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate
echo "Running python version $(python3 --version) from $(which python)"
echo "GPU info:"
nvidia-smi

# 3. Paths
PROJECT_PATH=/storage/slurm/schnackl/fakesync/cav-mae-sync
DATASET_JSON=/storage/slurm/schnackl/fakesync/data/fakeavceleb/FakeAVCeleb/preprocessed/dataset_info.json
SAVE_PATH=${PROJECT_PATH}/checkpoints/unsupervised_train_$(date +%Y%m%d_%H%M%S)

# 4. Training Configuration
# Note: batch_size is number of videos. Effective batch size = batch_size * total_frame
# With 64G VRAM, we can likely handle batch_size=8 or 16 with total_frame=16
BATCH_SIZE=8
TOTAL_FRAME=16
EPOCHS=50
LR=1e-4

echo "Starting training..."
echo "Project Path: ${PROJECT_PATH}"
echo "Save Path: ${SAVE_PATH}"

# 5. Run Training
python3 ${PROJECT_PATH}/train_unsupervised.py \
    --dataset_json ${DATASET_JSON} \
    --batch_size ${BATCH_SIZE} \
    --total_frame ${TOTAL_FRAME} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --save_path ${SAVE_PATH} \
    --contrast_loss_weight 0.01 \
    --mae_loss_weight 1.0 \
    --num_workers 8 \
    --log_freq 10 \
    --cls_token \
    --num_register_tokens 4

echo "Training finished at $(date) with exit code $?"
