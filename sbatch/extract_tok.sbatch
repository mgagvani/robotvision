#!/bin/bash
#SBATCH --job-name=extract_both
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/log/extract_both-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/log/extract_both-%j.err
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

srun python extract_planner_tok.py \
  --checkpoint camera-e2e-epoch=04-val_loss=2.90.ckpt \
  --data_dir /scratch/gilbreth/chang899/waymo_data/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --index_file index_train.pkl \
  --output_path planner_tokens_train.pt

srun python extract_planner_tok.py \
  --checkpoint camera-e2e-epoch=04-val_loss=2.90.ckpt \
  --data_dir /scratch/gilbreth/chang899/waymo_data/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --index_file index_val.pkl \
  --output_path planner_tokens_val.pt
