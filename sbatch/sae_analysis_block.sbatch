#!/bin/bash
#SBATCH --job-name=sae_analysis_block
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=06:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-analysis-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
cd "${PROJECT_DIR}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-10}"

RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${PROJECT_DIR}/camera-e2e-epoch=04-val_loss=2.90.ckpt}"
SPLIT="${SPLIT:-val}"
SAE_BLOCK="${SAE_BLOCK:-0}"
DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-4096}"
RELEVANT_SCENES_PER_FEATURE="${RELEVANT_SCENES_PER_FEATURE:-64}"
ALPHAS="${ALPHAS:-0,0.5,1.0,2.0}"

echo "Running block analysis bundle"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SAE_BLOCK=${SAE_BLOCK}"
echo "SPLIT=${SPLIT}"

python analyze_sae_intent.py \
    --run_root "${RUN_ROOT}" \
    --split "${SPLIT}" \
    --sae_block "${SAE_BLOCK}" \
    --device "${DEVICE}"

python analyze_sae_error.py \
    --run_root "${RUN_ROOT}" \
    --split "${SPLIT}" \
    --sae_block "${SAE_BLOCK}" \
    --device "${DEVICE}"

python analyze_sae_control.py \
    --run_root "${RUN_ROOT}" \
    --planner_checkpoint "${PLANNER_CHECKPOINT}" \
    --split "${SPLIT}" \
    --sae_block "${SAE_BLOCK}" \
    --device "${DEVICE}" \
    --encode_batch_size "${ENCODE_BATCH_SIZE}" \
    --relevant_scenes_per_feature "${RELEVANT_SCENES_PER_FEATURE}" \
    --alphas "${ALPHAS}"
