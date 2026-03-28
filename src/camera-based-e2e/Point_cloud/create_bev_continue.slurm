#!/bin/bash
#SBATCH --job-name=waymo_bev_cont
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --account=csso
#SBATCH --array=0-12

PYTHON=/scratch/gilbreth/kumar753/conda_envs/robo_env_310_new/bin/python
SCRIPT_DIR=/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e/Point_cloud
PARENT_DIR=/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e
DATA_DIR=/scratch/gilbreth/kumar753/robotvision/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0

export PYTHONPATH="${PARENT_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"
mkdir -p ${SCRIPT_DIR}/logs

TOTAL_TRAIN=251566
NUM_TASKS=13
CHUNK=$(( (TOTAL_TRAIN + NUM_TASKS - 1) / NUM_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

echo "Array task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}: frames ${START}..${START}+${CHUNK}"
echo "Already-processed .npy files will be skipped automatically."

$PYTHON ${SCRIPT_DIR}/create_bev.py \
    --data_dir $DATA_DIR \
    --split train \
    --start_idx $START \
    --n_items $CHUNK \
    --index_dir "${PARENT_DIR}" \
    --output_dir /scratch/gilbreth/kumar753/waymo_bev \
    --backend gpu
