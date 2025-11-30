#!/bin/bash
#SBATCH --job-name=extract_audio
#SBATCH --output=slurm_logs_flo/audio_%A_%a.out
#SBATCH --error=slurm_logs_flo/audio_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=0-7

# 1. Force stdout/stderr to flush immediately
export PYTHONUNBUFFERED=1

# 2. Safety: Set a default for CPUs if Slurm doesn't provide it
CPUS=${SLURM_CPUS_PER_TASK:-4}

echo "================ DEBUG INFO ================"
echo "Hostname: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs assigned: $CPUS"
echo "Date: $(date)"
echo "============================================"

# 3. Environment Setup
source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate

# Verify Python works before running the main script
which python
python --version

# 4. Library Paths (Keep your SOX setup)
export LD_LIBRARY_PATH="$HOME/sox/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$HOME/sox/usr/bin:$PATH"

# INPUT_CSV="/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb.csv"
INPUT_CSV="/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb_1percent.csv"
# TARGET_DIR="/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed/audio"
TARGET_DIR="/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed1pct/audio"
SCRIPT_PATH="/storage/slurm/schnackl/fakesync/cav-mae-sync/preprocess/extract_audio_new.py"
NUM_SHARDS=8  # Must match array count (0-15 is 16 items)

# Check if files exist
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input CSV not found at $INPUT_CSV"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found at $SCRIPT_PATH"
    exit 1
fi

echo "Starting Python Script..."

# 5. Run Python (Added -u for safety, removed srun)
python -u "$SCRIPT_PATH" \
    -input_file_list "$INPUT_CSV" \
    -target_fold "$TARGET_DIR" \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS \
    --num_workers $CPUS

echo "Python Script Finished with Exit Code: $?"