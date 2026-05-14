#!/bin/bash
#SBATCH --job-name=sae_train_block
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=08:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-train-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
cd "${PROJECT_DIR}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-10}"

RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
TOKENS_DIR="${TOKENS_DIR:-${RUN_ROOT}/tokens}"
SAE_OUTPUT_DIR="${SAE_OUTPUT_DIR:-${RUN_ROOT}/model}"
TRAIN_TOKEN_PATH="${TRAIN_TOKEN_PATH:-${TOKENS_DIR}/planner_tokens_train.pt}"
VAL_TOKEN_PATH="${VAL_TOKEN_PATH:-${TOKENS_DIR}/planner_tokens_val.pt}"
SAE_BLOCK="${SAE_BLOCK:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4096}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
K="${K:-auto}"
K_AUX="${K_AUX:-512}"
LAMBDA_AUX="${LAMBDA_AUX:-0.03125}"
DEAD_STEPS_THRESHOLD="${DEAD_STEPS_THRESHOLD:-500}"
GEOMETRIC_MEDIAN_SAMPLES="${GEOMETRIC_MEDIAN_SAMPLES:-10000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${SAE_OUTPUT_DIR}"

echo "Training SAE block ${SAE_BLOCK}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TRAIN_TOKEN_PATH=${TRAIN_TOKEN_PATH}"
echo "VAL_TOKEN_PATH=${VAL_TOKEN_PATH}"
echo "SAE_OUTPUT_DIR=${SAE_OUTPUT_DIR}"
echo "K=${K}"
echo "DEVICE=${DEVICE}"

python train_sae.py \
    --output_dir "${SAE_OUTPUT_DIR}" \
    --train_dataset "${TRAIN_TOKEN_PATH}" \
    --val_dataset "${VAL_TOKEN_PATH}" \
    --blocks "${SAE_BLOCK}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --lr "${LR}" \
    --max_epochs "${MAX_EPOCHS}" \
    --k "${K}" \
    --k_aux "${K_AUX}" \
    --lambda_aux "${LAMBDA_AUX}" \
    --dead_steps_threshold "${DEAD_STEPS_THRESHOLD}" \
    --geometric_median_samples "${GEOMETRIC_MEDIAN_SAMPLES}" \
    --grad_clip "${GRAD_CLIP}" \
    --device "${DEVICE}"
