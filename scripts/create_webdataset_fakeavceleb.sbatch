#!/bin/bash
#SBATCH --job-name="Create WebDataset FakeAVCeleb"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-webdataset-fav-%j.out
#SBATCH --error=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-webdataset-fav-%j.err

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "================ WebDataset Creation: FakeAVCeleb ================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "=================================================================="

# Environment
source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate
echo "Python: $(python3 --version) from $(which python3)"

# Paths — ADJUST THESE FOR YOUR SETUP
PROJECT_PATH=/storage/slurm/schnackl/fakesync/cav-mae-sync

# Use pre-generated identity-aware split from mm-dfd
# These CSVs have columns: video_name, target
TRAIN_CSV=/storage/slurm/schnackl/fakesync/data/fakeavceleb/fake_av_splits/train_split.csv
TEST_CSV=/storage/slurm/schnackl/fakesync/data/fakeavceleb/fake_av_splits/test_split.csv
OUTPUT_DIR=/storage/slurm/hunecke/fakesync/data/fakeavceleb/webdataset

# Processing config
MAX_AUDIO_LENGTH=1024
TOTAL_FRAME=16
IM_RES=224
NUM_MEL_BINS=128

echo "Train CSV: ${TRAIN_CSV}"
echo "Test CSV: ${TEST_CSV}"
echo "Output: ${OUTPUT_DIR}"

# Create train shards
echo "Creating train shards..."
python3 ${PROJECT_PATH}/preprocess/create_webdataset.py \
    --input_file ${TRAIN_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --split_name train \
    --total_frame ${TOTAL_FRAME} \
    --im_res ${IM_RES} \
    --max_audio_length ${MAX_AUDIO_LENGTH} \
    --num_mel_bins ${NUM_MEL_BINS} \
    --num_workers 16

# Create test shards
echo "Creating test shards..."
python3 ${PROJECT_PATH}/preprocess/create_webdataset.py \
    --input_file ${TEST_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --split_name test \
    --total_frame ${TOTAL_FRAME} \
    --im_res ${IM_RES} \
    --max_audio_length ${MAX_AUDIO_LENGTH} \
    --num_mel_bins ${NUM_MEL_BINS} \
    --num_workers 16

echo "WebDataset creation finished at $(date) with exit code $?"
