#!/bin/bash
#SBATCH --job-name="Preprocess CAV-MAE Shards"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-preprocess-%j.out
#SBATCH --error=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-preprocess-%j.err
#SBATCH --partition=batch

# 1. Force stdout/stderr to flush immediately
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "================ DEBUG INFO ================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# 2. Environment Setup
source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate
echo "Running python version $(python3 --version) from $(which python)"

# 3. Paths
PROJECT_PATH=/storage/slurm/schnackl/fakesync/cav-mae-sync
# Adjusted input/output for likely cluster location, user can check
INPUT_FILE=${PROJECT_PATH}/datafiles/all_unsupervised.csv
OUTPUT_DIR=/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed/shards_train_v2

# 4. Configuration
NUM_SHARDS=200
WORKERS=32
# Legacy Parity Configs
MAX_AUDIO_LENGTH=1024
TARGET_LENGTH=416
TOTAL_FRAME=16
IM_RES=224

echo "Starting extraction..."
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_DIR}"

# 5. Run Extraction
srun python3 ${PROJECT_PATH}/preprocess/create_sharded_dataset.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --num_shards ${NUM_SHARDS} \
    --num_workers ${WORKERS} \
    --max_audio_length ${MAX_AUDIO_LENGTH} \
    --target_length ${TARGET_LENGTH} \
    --total_frame ${TOTAL_FRAME} \
    --im_res ${IM_RES}

echo "Preprocessing finished at $(date) with exit code $?"
