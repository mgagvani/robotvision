#!/bin/bash


#SBATCH --time=04:00:00
#SBATCH --partition=v100
#SBATCH --account=csso
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10

module load cuda anaconda
conda activate /home/bnamikas/.conda/envs/2025.06-py313/python3.10
srun python train.py --data_dir /scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0