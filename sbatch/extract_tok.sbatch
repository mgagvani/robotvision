#!/bin/bash
#SBATCH --job-name=sae_extract_tok
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=06:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-extract-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
cd "${PROJECT_DIR}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-10}"
export HF_HOME="${HF_HOME:-/scratch/negishi/mgagvani/hfcache}"
export WAYMO_DATA_DIR="${WAYMO_DATA_DIR:-/scratch/negishi/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0}"

RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
TOKENS_DIR="${TOKENS_DIR:-${RUN_ROOT}/tokens}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/camera-e2e-epoch=04-val_loss=2.90.ckpt}"
SPLIT="${SPLIT:-train}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
N_WORKERS="${N_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"

case "${SPLIT}" in
    train)
        INDEX_FILE_DEFAULT="index_train.pkl"
        ;;
    val)
        INDEX_FILE_DEFAULT="index_val.pkl"
        ;;
    *)
        echo "Unsupported SPLIT=${SPLIT}. Expected train or val." >&2
        exit 1
        ;;
esac

INDEX_FILE="${INDEX_FILE:-${INDEX_FILE_DEFAULT}}"
OUTPUT_PATH="${OUTPUT_PATH:-${TOKENS_DIR}/planner_tokens_${SPLIT}.pt}"

mkdir -p "${TOKENS_DIR}"

echo "Running token extraction"
echo "SPLIT=${SPLIT}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "INDEX_FILE=${INDEX_FILE}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "DEVICE=${DEVICE}"

CMD=(
    python extract_planner_tok.py
    --checkpoint "${CHECKPOINT_PATH}"
    --data_dir "${WAYMO_DATA_DIR}"
    --index_file "${INDEX_FILE}"
    --batch_size "${EXTRACT_BATCH_SIZE}"
    --nw "${N_WORKERS}"
    --device "${DEVICE}"
    --output_path "${OUTPUT_PATH}"
)

if [[ -n "${N_ITEMS:-}" ]]; then
    CMD+=(--n_items "${N_ITEMS}")
fi

"${CMD[@]}"
