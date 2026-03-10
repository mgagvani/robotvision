#!/bin/bash


#SBATCH --time=10:00:00
#SBATCH --partition=a10
#SBATCH --account=csso
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

module load cuda/12.6.0 anaconda
conda activate /home/bnamikas/.conda/envs/2025.06-py313/python3.10

# Avoid stale submit-shell values (e.g., from an interactive allocation)
# conflicting with this job's cpu=8 request.
unset SLURM_CPUS_PER_TASK

srun python train_diffuse_rework_reg_copy.py --data_dir /scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0
