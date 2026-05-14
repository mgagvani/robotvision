#!/bin/bash
#SBATCH --job-name=sae_control
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=06:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-control-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${PROJECT_DIR}/camera-e2e-epoch=04-val_loss=2.90.ckpt}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"
SAE_BLOCK="${SAE_BLOCK:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/analysis/block_${SAE_BLOCK}}"
RELEVANT_SCENES_PER_FEATURE="${RELEVANT_SCENES_PER_FEATURE:-64}"
ALPHAS="${ALPHAS:-0,0.5,1.0,2.0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-4096}"

cd "${PROJECT_DIR}"

echo "Running SAE control analysis"
echo "RUN_ROOT=${RUN_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT}"
echo "DEVICE=${DEVICE}"
echo "SPLIT=${SPLIT}"
echo "SAE_BLOCK=${SAE_BLOCK}"
echo "RELEVANT_SCENES_PER_FEATURE=${RELEVANT_SCENES_PER_FEATURE}"
echo "ALPHAS=${ALPHAS}"

python analyze_sae_control.py \
    --run_root "${RUN_ROOT}" \
    --planner_checkpoint "${PLANNER_CHECKPOINT}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --sae_block "${SAE_BLOCK}" \
    --device "${DEVICE}" \
    --encode_batch_size "${ENCODE_BATCH_SIZE}" \
    --relevant_scenes_per_feature "${RELEVANT_SCENES_PER_FEATURE}" \
    --alphas "${ALPHAS}"
