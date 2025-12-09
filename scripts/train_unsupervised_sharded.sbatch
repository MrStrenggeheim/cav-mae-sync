#!/bin/bash
#SBATCH --job-name="Train CAV-MAE-Sync Sharded"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=VRAM:32G
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-train-sharded-%j.out
#SBATCH --error=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-train-sharded-%j.err

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
SHARDED_DATASET_DIR=/storage/slurm/hunecke/fakesync/data/voxceleb2/preprocessed/shards_train
SAVE_PATH=${PROJECT_PATH}/outputs/checkpoints/unsupervised_train_sharded_$(date +%Y%m%d_%H%M%S)

# 4. Training Configuration
# With sharded dataset and IterableDataset, BATCH_SIZE is per-worker batch size if we aren't careful, 
# but PyTorch DataLoader usually treats batch_size as global batch size or per-worker depending on distributed.
# In single-node non-DDP, standard DataLoader batch_size is total batch size.
BATCH_SIZE=8
TOTAL_FRAME=16
EPOCHS=10
WARMUP_EPOCHS=1
LR=1e-4

echo "Starting training..."
echo "Project Path: ${PROJECT_PATH}"
echo "Sharded Dataset Dir: ${SHARDED_DATASET_DIR}"
echo "Save Path: ${SAVE_PATH}"

# 5. Run Training
# Note: removed --dataset_json and added --sharded_dataset_dir
# Note: --contrastive_heads is NOT used because the pretrained model (cav_mae_sync.pth) 
#       was trained without contrastive heads. Using them would create architecture mismatch.
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
    --contrast_loss_weight 0.1 \
    --mae_loss_weight 1.0 \
    --log_freq 10 \
    --cls_token \
    --num_register_tokens 8 \
    # --fast_dev_run \

echo "Training finished at $(date) with exit code $?"
