#!/bin/bash


#SBATCH --time=48:00:00
#SBATCH --partition=a10
#SBATCH --account=csso
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/12.6.0 anaconda
conda activate /home/bnamikas/.conda/envs/2025.06-py313/python3.10
srun python compute.py --index_file index_val.pkl --data_dir /scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0 --output_dir /scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/precomputed/val
