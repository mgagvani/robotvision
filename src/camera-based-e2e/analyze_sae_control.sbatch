#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/slurm_logs/analyze-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/slurm_logs/analyze-%j.err

#SBATCH --account=labi
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf
cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e || exit 1

python analyze_sae_control.py \
  --data_path output/extract_train_control.pt \
  --sae_ckpt output/sae_control_pred.pt \
  --output_dir output/analysis_control_small \
  --top_k 15