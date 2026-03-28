#!/bin/bash
#SBATCH --job-name=waymo_occ_val
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --account=csso
#SBATCH --array=0-1

PYTHON=/scratch/gilbreth/kumar753/conda_envs/robo_env_310_new/bin/python
SCRIPT_DIR=/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e/Point_cloud
PARENT_DIR=/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e
DATA_DIR=/scratch/gilbreth/kumar753/robotvision/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0

export PYTHONPATH="${PARENT_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"
mkdir -p ${SCRIPT_DIR}/logs

TOTAL=10000
NUM_TASKS=2
CHUNK=$(( (TOTAL + NUM_TASKS - 1) / NUM_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

echo "OCC val_new task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}: frames ${START}..${START}+${CHUNK}"

$PYTHON ${SCRIPT_DIR}/create_occ_labels.py \
    --data_dir  "$DATA_DIR" \
    --split     val \
    --index_dir "${PARENT_DIR}" \
    --start_idx "$START" \
    --n_items   "$CHUNK" \
    --output_dir /scratch/gilbreth/kumar753/waymo_occ_new \
    --device cuda
