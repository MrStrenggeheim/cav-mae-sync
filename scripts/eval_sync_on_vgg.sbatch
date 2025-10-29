#!/bin/bash
#SBATCH --job-name="Forward CAV-MAE-Sync"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
# SBATCH --gres=gpu:2,VRAM:14G
# SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:TURING"

#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-%j.out
#SBATCH --error=/storage/slurm/schnackl/fakesync/slurm_logs_flo/slurm-cav-mae-sync-%j.err

# run cav-mae pretraining, fits larger GPUs (4*24GB GPUs)

echo "LDD version $(ldd --version)"
source /storage/slurm/schnackl/fakesync/myVenv/.venv/bin/activate
echo "Running python version $(python3 --version) from $(which python)"
echo "GPU info:"
nvidia-smi

# CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 -W ignore /storage/user/hunecke/fakesync/NewAVFF/eval.py \
	# --checkpoint ${model_path} --csv_file ${data} --output ${output_path} --save_full_path



project_path=/storage/slurm/schnackl/fakesync/cav-mae-sync
model_path=/storage/slurm/schnackl/fakesync/cav-mae-sync/pretrained_models/cav_mae_sync.pth

python3 ${project_path}/src/activation_eval.py \
    --model_path ${model_path} \
    --csv_path /storage/slurm/schnackl/fakesync/data/vgg_test/preprocessed/dataset_info.csv \
    --json_path /storage/slurm/schnackl/fakesync/data/vgg_test/preprocessed/dataset_info.json \
    --batch_size 16 \
    --output ${project_path}/outputs/vgg_test/ \
    --max_samples 16 ##8192 \ 48 batch