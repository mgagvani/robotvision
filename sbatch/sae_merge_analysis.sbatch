#!/bin/bash
#SBATCH --job-name=sae_merge_analysis
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=01:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-merge-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
cd "${PROJECT_DIR}"

RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_ROOT}/analysis}"

echo "Merging analysis outputs"
echo "RUN_ROOT=${RUN_ROOT}"
echo "ANALYSIS_DIR=${ANALYSIS_DIR}"

python merge_sae_block_analysis.py \
    --run_root "${RUN_ROOT}" \
    --analysis_dir "${ANALYSIS_DIR}"
