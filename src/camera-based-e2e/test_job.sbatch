#!/bin/bash
#SBATCH --job-name=waymo-e2e-test
#SBATCH --output=waymo-e2e-test-%j.out
#SBATCH --error=waymo-e2e-test-%j.err

#SBATCH --account=csso
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load anaconda
source activate /scratch/gilbreth/shar1159/conda_envs/waymo

cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e

# (optional but helpful)
export PYTHONUNBUFFERED=1

srun python testing.py \
  --data_dir /scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0/ \
  --batch_size 16 \
  --ckpt /scratch/gilbreth/shar1159/checkpoints/camera-e2e-epoch=02-val_loss=1.67.ckpt

