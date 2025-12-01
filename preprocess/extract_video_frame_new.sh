#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --output=slurm_logs_flo/extract_frames_%A_%a.out
#SBATCH --error=slurm_logs_flo/extract_frames_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-1  # 16 jobs total

# 1. Force stdout/stderr to flush immediately
# export PYTHONUNBUFFERED=1

# 2. Safety: Set a default for CPUs if Slurm doesn't provide it
CPUS=${SLURM_CPUS_PER_TASK:-4}

echo "================ DEBUG INFO ================"
echo "Hostname: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs assigned: $CPUS"
echo "Date: $(date)"
echo "============================================"

# 2. Environment Setup
source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate

# Verify Python works before running the main script
which python
python --version

# 3. Config
INPUT_CSV="/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb.csv"
# INPUT_CSV="/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_1percent.csv"
TARGET_DIR="/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed/frames"
# TARGET_DIR="/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed1pct/frames"
SCRIPT_PATH="/storage/slurm/schnackl/fakesync/cav-mae-sync/preprocess/extract_video_frame_new.py"

NUM_SHARDS=2  # Must match the count of --array (0-7 = 8)
FRAMES_TO_EXTRACT=16

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID on $(hostname)"

# 5. Run Python
python -u "$SCRIPT_PATH" \
    -input_file_list "$INPUT_CSV" \
    -target_fold "$TARGET_DIR" \
    --extract_frame_num $FRAMES_TO_EXTRACT \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS \
    --num_workers $CPUS

echo "Python Script Finished with Exit Code: $?"