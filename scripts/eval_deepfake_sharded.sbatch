#!/bin/bash
#SBATCH --job-name="Eval CAV-MAE-Sync DeepFake Sharded"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/hunecke/fakesync/slurm_logs/eval/slurm-cav-mae-sync-eval-sharded-%j.out
#SBATCH --error=/storage/slurm/hunecke/fakesync/slurm_logs/eval/slurm-cav-mae-sync-eval-sharded-%j.err

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
# Sharded eval dataset (preprocessed with create_sharded_dataset.py)
SHARDED_EVAL_DIR=/storage/slurm/hunecke/fakesync/data/fakeavceleb/preprocessed/shards_eval

CHECKPOINT_PATH=${PROJECT_PATH}/outputs/checkpoints/unsupervised_train_sharded/best.ckpt
OUTPUT_DIR=${PROJECT_PATH}/outputs/eval_sharded_$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV=${OUTPUT_DIR}/eval_results.csv

# 4. Evaluation Configuration
BATCH_SIZE=1
TOTAL_FRAME=16
TARGET_LENGTH=48

# Normalization stats (should match training)
MEAN=-4.050048828125
STD=4.067018032073975

echo "Starting evaluation..."
echo "Project Path: ${PROJECT_PATH}"
echo "Sharded Eval Dir: ${SHARDED_EVAL_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# 5. Run Evaluation
python3 ${PROJECT_PATH}/eval_deepfake.py \
    --sharded_dataset_dir ${SHARDED_EVAL_DIR} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --output_csv ${OUTPUT_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --total_frame ${TOTAL_FRAME} \
    --target_length ${TARGET_LENGTH} \
    --num_workers 8 \
    --mean ${MEAN} \
    --std ${STD} \
    --no_masking
    # Optional: --label_csv /path/to/labels.csv for AUC-ROC

echo "Evaluation finished at $(date) with exit code $?"
echo "Results saved to ${OUTPUT_CSV}"
echo "TensorBoard logs in ${OUTPUT_DIR}/tensorboard_eval"
