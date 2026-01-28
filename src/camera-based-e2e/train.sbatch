#!/bin/bash
#SBATCH --job-name=monocular_train
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --account=csso
#SBATCH --partition=a10

# Load modules
module load conda

# Activate conda environment (try both methods)
source ~/.bashrc 2>/dev/null || true
conda activate waymo_dataloader || source activate waymo_dataloader

# Change to script directory
cd /home/mathur91/robotvision/src/camera-based-e2e

# Run training
python train.py \
    --data_dir /scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0/ \
    --batch_size 16 \
    --lr 0.001 \
    --max_epochs 10

echo "Training completed at $(date)"

