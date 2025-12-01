#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --output=slurm_logs_flo/extract_%A_%a.out
#SBATCH --error=slurm_logs_flo/extract_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --array=0-14

# 1. Config
INPUT_CSV="/storage/slurm/schnackl/fakesync/data/voxceleb2/voxceleb2_dataset_split_without_fakeavceleb.csv"
FINAL_DEST="/storage/slurm/schnackl/fakesync/data/voxceleb2/preprocessed/frames"
SCRIPT_PATH="/storage/slurm/schnackl/fakesync/cav-mae-sync/preprocess/extract_video_frame_new.py"


NUM_FRAMES=16

# 2. Local Scratch
BASE_TMP=${SLURM_TMPDIR:-/tmp}
SCRATCH_DIR="$BASE_TMP/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID"
mkdir -p "$SCRATCH_DIR"

source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CV_NUM_THREADS=0

# 3. Run Python (High Speed NVMe)
echo "Step 1: Extracting frames to NVMe..."
python -u "$SCRIPT_PATH" \
    -input_file_list "$INPUT_CSV" \
    -target_fold "$SCRATCH_DIR" \
    --extract_frame_num $NUM_FRAMES \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards 15 \
    --num_workers 6

PYTHON_EXIT=$?

# 4. The Parallel Sync with Progress Monitor
if [ $PYTHON_EXIT -eq 0 ]; then
    echo "Step 2: Starting Parallel Sync..."
    mkdir -p "$FINAL_DEST"
    
    # --- A. Launch 16 Background Rsyncs ---
    # This bypasses the speed limit by opening 16 connections at once.
    # We redirect output to /dev/null because we will track the PIDs instead.
    pids=""
    for i in $(seq 0 $((NUM_FRAMES - 1))); do
        mkdir -p "$FINAL_DEST/frame_$i"
        rsync -aW --no-compress --remove-source-files \
            "$SCRATCH_DIR/frame_$i/" "$FINAL_DEST/frame_$i/" > /dev/null 2>&1 &
        pids="$pids $!"
    done

    # --- B. Progress Monitor Loop ---
    # This prevents the log flood and shows a clean progress bar.
    total_jobs=$NUM_FRAMES
    echo "Syncing $total_jobs frame folders in parallel..."
    
    while true; do
        # Count how many of our rsync PIDs are still running
        running=0
        for pid in $pids; do
            if kill -0 $pid 2>/dev/null; then
                running=$((running + 1))
            fi
        done
        
        # Calculate percentage
        finished=$((total_jobs - running))
        percent=$((finished * 100 / total_jobs))
        
        # Log update
        echo "Sync Progress: $percent% ($finished/$total_jobs folders finished)"
        
        # Break if all done
        if [ $running -eq 0 ]; then
            break
        fi
        
        # Check every 10 seconds
        sleep 10
    done
    
    echo "Sync Complete."
    rm -rf "$SCRATCH_DIR"
else
    echo "Python Failed. Aborting."
    exit $PYTHON_EXIT
fi