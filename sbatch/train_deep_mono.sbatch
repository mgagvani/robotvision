#!/bin/bash
#SBATCH --job-name=train_deep_mono
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/log/deep_mono-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/log/deep_mono-%j.err
#
#SBATCH --account=labi
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=84:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf

cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e

mkdir -p log

srun python train.py \
  --data_dir /scratch/gilbreth/chang899/waymo_data