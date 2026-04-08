#!/bin/bash
#SBATCH --job-name="CAV-MAE Test Suite"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-test-all-%j.out
#SBATCH --error=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-test-all-%j.err

export PYTHONUNBUFFERED=1

echo "================ CAV-MAE Sync Test Suite ================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "GPU:"
nvidia-smi
echo "========================================================="

# Environment — adjust to your venv
source /storage/slurm/hunecke/fakesync/.venv/bin/activate
echo "Python: $(python3 --version) from $(which python3)"

# Ensure webdataset is installed
pip install webdataset --quiet 2>/dev/null

# Project path
PROJECT_PATH=/storage/slurm/hunecke/fakesync/cav-mae-sync

echo ""
echo "Running comprehensive test suite..."
echo ""

python3 ${PROJECT_PATH}/tests/test_all_changes.py

EXIT_CODE=$?
echo ""
echo "Test suite finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
