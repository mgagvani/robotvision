#!/bin/bash
#SBATCH --job-name=sae-b3-accel
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/slurm_logs/intervene-b3-accel-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/slurm_logs/intervene-b3-accel-%j.err

#SBATCH --account=ziran
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=5

module load anaconda
conda activate /scratch/gilbreth/chang899/conda_envs/lead_ltf

set -euo pipefail

cd /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e || exit 1

export MPLCONFIGDIR="/tmp/matplotlib_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID}"
export FC_CACHEDIR="${XDG_CACHE_HOME}/fontconfig"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${FC_CACHEDIR}"
mkdir -p /scratch/gilbreth/chang899/codes/int/slurm_logs
mkdir -p sae_output/intervention_accel_latent_block3

python intervene_sae_accel_latent.py \
  --model_ckpt /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/camera-e2e-epoch=04-val_loss=2.90.ckpt \
  --sae_ckpt /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/sae_checkpoints/sae_block_3.pt \
  --planner_sae_block_index 3 \
  --flip_behaviors \
  --output_dir /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/sae_output/intervention_accel_latent_block3 \
  --target_accel_mps2 2.0 \
  --template_scene_count 64 \
  --speed_delta_thresh 0.5 \
  --n_items 512 \
  --batch_size 16 \
  --max_plots 24
