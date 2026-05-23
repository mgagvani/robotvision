#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/slurm_logs/analyze-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/slurm_logs/analyze-%j.err

#SBATCH --account=ziran
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf
cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e || exit 1

python analyze_sae_coactivation.py \
  --data_path output/extract_train_control.pt \
  --sae_ckpt output/sae_control_pred.pt \
  --output_dir output/coactivation_control \
  --data_dir /scratch/gilbreth/chang899/waymo_data/waymo_open_dataset_end_to_end_camera_v_1_0_0
