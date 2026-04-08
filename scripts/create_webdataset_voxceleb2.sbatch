#!/bin/bash
#SBATCH --job-name="Create WebDataset VoxCeleb2"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-webdataset-vox2-%j.out
#SBATCH --error=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-webdataset-vox2-%j.err

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "================ WebDataset Creation: VoxCeleb2 ================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "================================================================"

# Environment
source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate
echo "Python: $(python3 --version) from $(which python3)"

# Paths — ADJUST THESE FOR YOUR SETUP
PROJECT_PATH=/storage/slurm/schnackl/fakesync/cav-mae-sync
INPUT_FILE=/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_all.csv
OUTPUT_DIR=/storage/slurm/hunecke/fakesync/data/voxceleb2/webdataset

# Identity-aware splitting:
# 1. Exclude FakeAVCeleb identities from VoxCeleb2 training
# 2. Split remaining by identity 90/5/5 (train/val/test)
EXCLUDE_IDENTITIES=/storage/slurm/schnackl/fakesync/data/fakeavceleb_identities.txt

# Processing config (match training params)
MAX_AUDIO_LENGTH=1024
TOTAL_FRAME=16
IM_RES=224
NUM_MEL_BINS=128
MAX_SHARD_SIZE=1000000000  # ~1GB per shard

echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Exclude identities: ${EXCLUDE_IDENTITIES}"

srun python3 ${PROJECT_PATH}/preprocess/create_webdataset.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --split_by_identity \
    --split_ratios 0.9 0.05 0.05 \
    --exclude_identities_file ${EXCLUDE_IDENTITIES} \
    --total_frame ${TOTAL_FRAME} \
    --im_res ${IM_RES} \
    --max_audio_length ${MAX_AUDIO_LENGTH} \
    --num_mel_bins ${NUM_MEL_BINS} \
    --max_shard_size ${MAX_SHARD_SIZE} \
    --num_workers 32 \
    --seed 42

echo "WebDataset creation finished at $(date) with exit code $?"
