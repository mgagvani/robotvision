#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm_logs/eval-%j.out
#SBATCH --error=slurm_logs/eval-%j.err

#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5

module load anaconda
conda activate /scratch/gilbreth/shar1159/conda_envs/waymo
cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e || exit 1

python eval_sae.py