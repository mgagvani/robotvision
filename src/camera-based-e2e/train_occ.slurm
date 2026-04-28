#!/bin/bash
#SBATCH --job-name=waymo_occ_10ep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --account=csso

PYTHON=/scratch/gilbreth/kumar753/conda_envs/robo_env_310_new/bin/python
SCRIPT_DIR=/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e
DATA_DIR=/scratch/gilbreth/kumar753/robotvision/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0
OCC_ROOT=/scratch/gilbreth/kumar753/waymo_occ_new

mkdir -p ${SCRIPT_DIR}/logs
cd ${SCRIPT_DIR}

$PYTHON train_occ.py \
  --data_dir "$DATA_DIR" \
  --occ_root "$OCC_ROOT" \
  --batch_size 4 \
  --lr 5e-5 \
  --max_epochs 10

