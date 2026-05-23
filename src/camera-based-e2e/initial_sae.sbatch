#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/logs/sae-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/logs/sae-%j.err

#SBATCH --account=labi
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf
cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e || exit 1

#python models/initial_sae.py --data output/extract_train_score.pt --mode score --ckpt output/sae_score_in.pt
#python models/initial_sae.py --data output/extract_train_traj.pt --mode traj --ckpt output/sae_traj_feat.pt
python models/initial_sae.py --data output/extract_train_control.pt --mode control --ckpt output/sae_control_pred.pt
