#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/slurm_logs/extract-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/slurm_logs/extract-%j.err

#SBATCH --account=labi
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf
cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e || exit 1

python extract_queries_by_intent.py