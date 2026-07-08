#!/bin/bash
#SBATCH --job-name=camera_e2e_train
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=24:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/exp/slurm-train-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-amdgpu}"

REPO_DIR="${REPO_DIR:-/home/mgagvani/robotvision}"
PROJECT_DIR="${REPO_DIR}/src/camera-based-e2e"
SCRIPT_PATH="${PROJECT_DIR}/train.py"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/exp"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-10}"
export HF_HOME="${HF_HOME:-/scratch/negishi/mgagvani/hfcache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="${DATASET:-waymo}"
DATA_DIR="${DATA_DIR:-${WAYMO_DATA_DIR:-/scratch/negishi/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"

ARGS=(
  --data_dir "${DATA_DIR}"
  --dataset "${DATASET}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --max_epochs "${MAX_EPOCHS}"
)

if [[ "${COMPILE:-0}" == "1" ]]; then
  ARGS+=(--compile)
fi

if [[ "${PROFILE:-0}" == "1" ]]; then
  ARGS+=(--profile)
fi

echo "Launching single-GPU camera E2E training"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATASET=${DATASET}"
echo "DATA_DIR=${DATA_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "MAX_EPOCHS=${MAX_EPOCHS}"

python "${SCRIPT_PATH}" "${ARGS[@]}"
