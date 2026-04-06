#!/bin/bash
#SBATCH --job-name=train_sae
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/log/train_sae-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/log/train_sae-%j.err
#
#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=84:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf

cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e

mkdir -p log

srun python train_sae.py \
  --output_dir /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output \
  --train_dataset planner_tokens_train.pt \
  --val_dataset planner_tokens_val.pt

