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
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
cd "${PROJECT_DIR}"

# Optimizations for protobuf and per-rank CPU threading.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-10}"
export HF_HOME="${HF_HOME:-/scratch/negishi/mgagvani/hfcache}"

# Avoid torchrun default rendezvous port collisions on shared nodes.
export MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))

# Faster of the two WOD copies benchmarked on 2026-03-28.
export WAYMO_DATA_DIR="/scratch/negishi/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0"

NPROC_PER_NODE=1

# Single-GPU default: global batch size 128.
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"

echo "Running camera-based-e2e training from ${PROJECT_DIR}"
echo "WAYMO_DATA_DIR=${WAYMO_DATA_DIR}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
echo "effective_global_batch_size=$((NPROC_PER_NODE * PER_GPU_BATCH_SIZE))"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train.py \
    --data_dir "${WAYMO_DATA_DIR}" \
    --batch_size "${PER_GPU_BATCH_SIZE}" \
    --lr 1e-4 \
    --max_epochs 10 \
    --dataset waymo \
#     --profile

# To override the default batch size, submit with:
# PER_GPU_BATCH_SIZE=64 sbatch train.sbatch
