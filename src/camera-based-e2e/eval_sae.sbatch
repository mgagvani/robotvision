#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm_logs/eval-%j.out
#SBATCH --error=slurm_logs/eval-%j.err

#SBATCH --account=ziran
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf
cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e || exit 1

python eval_sae.py