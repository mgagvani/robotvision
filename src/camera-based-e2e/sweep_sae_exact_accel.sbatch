#!/bin/bash
#SBATCH --job-name=sae-accel-sweep
#SBATCH --output=/scratch/gilbreth/chang899/codes/int/slurm_logs/sweep-exact-accel-%j.out
#SBATCH --error=/scratch/gilbreth/chang899/codes/int/slurm_logs/sweep-exact-accel-%j.err

#SBATCH --account=ziran
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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
mkdir -p output/intervention_exact_accel_sweep

python sweep_sae_exact_accel.py \
  --targets -4 -2 0 2 4 6 \
  --model_ckpt /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/camera-e2e-epoch=04-val_loss=2.90.ckpt \
  --sae_ckpt /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output/sae_control_pred.pt \
  --output_root /scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/output/intervention_exact_accel_sweep \
  --max_plots 0
