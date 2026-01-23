#!/bin/bash
#SBATCH --job-name="Eval CAV-MAE-Sync DeepFake"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/hunecke/fakesync/slurm_logs/eval/slurm-cav-mae-sync-eval-%j.out
#SBATCH --error=/storage/slurm/hunecke/fakesync/slurm_logs/eval/slurm-cav-mae-sync-eval-%j.err

# 1. Force stdout/stderr to flush immediately
export PYTHONUNBUFFERED=1

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
# TODO: Update this to the specific dataset json you want to evaluate on (e.g. test set)
# DATASET_JSON=/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed/voxceleb2_dataset_split_without_fakeavceleb_test_dataset_info.json
# DATASET_JSON=/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed1pct/voxceleb2_dataset_split_without_fakeavceleb_1percent_dataset_info.json
# eval on fakeavceleb:
DATASET_JSON=/storage/slurm/schnackl/fakesync/data/fakeavceleb/FakeAVCeleb/preprocessed/dataset_info.json

# TODO: Update this to your trained checkpoint path
# Example: ${PROJECT_PATH}/checkpoints/unsupervised_train_20231027_120000/epoch=49-step=1000.ckpt
CHECKPOINT_PATH=${PROJECT_PATH}/outputs/checkpoints/unsupervised_voxceleb2_1epoch_1worker/last.ckpt

OUTPUT_CSV=${PROJECT_PATH}/outputs/checkpoints/unsupervised_voxceleb2_1epoch_1worker/eval_results_$(date +%Y%m%d_%H%M%S).csv

# 4. Evaluation Configuration
# Batch size must be 1 for per-video scoring
BATCH_SIZE=1
TOTAL_FRAME=16

# Normalization stats
# Default values:
MEAN=-4.050048828125
STD=4.067018032073975

echo "Starting evaluation..."
echo "Project Path: ${PROJECT_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output CSV: ${OUTPUT_CSV}"

# Create output directory
mkdir -p $(dirname ${OUTPUT_CSV})

# 5. Run Evaluation
python3 ${PROJECT_PATH}/eval_deepfake.py \
    --dataset_json ${DATASET_JSON} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --output_csv ${OUTPUT_CSV} \
    --batch_size ${BATCH_SIZE} \
    --total_frame ${TOTAL_FRAME} \
    --num_workers 4 \
    --mean ${MEAN} \
    --std ${STD} \
    --no_masking

echo "Evaluation finished at $(date) with exit code $?"
