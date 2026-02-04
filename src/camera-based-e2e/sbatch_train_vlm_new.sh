#!/usr/bin/env bash
#SBATCH --job-name=vlm_new_train
#SBATCH --partition=v100
#SBATCH --account=csso
#SBATCH --time=17:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=165G
#SBATCH --output=logs/slurm_vlm_new_%j.out
#SBATCH --error=logs/slurm_vlm_new_%j.err

: "${DATA_DIR:=/scratch/gilbreth/$USER/waymo-data}"

# Submit from project root: sbatch src/camera-based-e2e/sbatch_train_vlm_new.sh
cd "${SLURM_SUBMIT_DIR:?}"
[[ -d src/camera-based-e2e ]] && cd src/camera-based-e2e
mkdir -p logs
# Project root: where scripts/ lives (one level up when we're in src/camera-based-e2e)
ROOT="$(pwd)"
[[ -f scripts/download.sh ]] || ROOT="$(cd ../../ && pwd)"

# Conda: load .bashrc so "conda init" runs; or set CONDA_BASE before sbatch
[[ -n "${CONDA_BASE:-}" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
source ~/.bashrc 2>/dev/null || true
conda activate "${CONDA_ENV:-rv}"

export HF_HOME="${HF_HOME:-/scratch/gilbreth/$USER/hfcache}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Run download script (from project root) and capture its job ID
# DOWNLOAD_JOB=$(sbatch --parsable "$ROOT/scripts/download.sh" "$DATA_DIR")
# if [[ -z "$DOWNLOAD_JOB" ]]; then
#     echo "Failed to submit download job." >&2
#     exit 1
# fi
# echo "Download job ID: $DOWNLOAD_JOB. Waiting for it to finish..."
# while squeue -j "$DOWNLOAD_JOB" 2>/dev/null | grep -q "$DOWNLOAD_JOB"; do
#     sleep 30
# done
# EXIT_CODE=$(sacct -j "$DOWNLOAD_JOB" -n -o ExitCode 2>/dev/null | head -1 | tr -d ' ')
# if [[ -n "$EXIT_CODE" && "$EXIT_CODE" != "0:0" ]]; then
#     echo "Download job failed (exit $EXIT_CODE). Exiting." >&2
#     exit 1
# fi
# echo "Download job completed. Starting training..."

python vlm_new/train.py --data_dir "$DATA_DIR"
