#!/bin/bash
#SBATCH --job-name="Preprocess CAV-MAE Shards"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/hunecke/fakesync/slurm_logs/pre/fakeavceleb-preprocess-%j.out
#SBATCH --error=/storage/slurm/hunecke/fakesync/slurm_logs/pre/fakeavceleb-preprocess-%j.err

# 1. Force stdout/stderr to flush immediately
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "================ DEBUG INFO ================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# 2. Environment Setup
source /storage/slurm/hunecke/fakesync/.venv/bin/activate
echo "Running python version $(python3 --version) from $(which python)"

# 3. Paths
PROJECT_PATH=/storage/slurm/hunecke/fakesync/cav-mae-sync
# Adjusted input/output for likely cluster location, user can check
INPUT_FILE=/storage/slurm/schnackl/fakesync/data/voxtest_fakeavceleb/voxtest_fakeavceleb.csv
OUTPUT_DIR=/storage/slurm/hunecke/fakesync/data/voxtest_fakeavceleb/preprocessed/shards

# 4. Configuration
NUM_SHARDS=10
WORKERS=16
# Legacy Parity Configs
MAX_AUDIO_LENGTH=1024
TARGET_LENGTH=42
TOTAL_FRAME=16
IM_RES=224

echo "Starting extraction..."
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_DIR}"

# 5. Run Extraction
python3 ${PROJECT_PATH}/preprocess/create_sharded_dataset.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --num_shards ${NUM_SHARDS} \
    --num_workers ${WORKERS} \
    --max_audio_length ${MAX_AUDIO_LENGTH} \
    --target_length ${TARGET_LENGTH} \
    --total_frame ${TOTAL_FRAME} \
    --im_res ${IM_RES}

echo "Preprocessing finished at $(date) with exit code $?"
