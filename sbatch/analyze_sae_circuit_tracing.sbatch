#!/bin/bash
#SBATCH --job-name=sae_circuit
#SBATCH --output=log/sae_circuit-%j.out
#SBATCH --error=log/sae_circuit-%j.err
#
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10

module load anaconda
conda activate "${CONDA_ENV:-/home/mgagvani/.conda/envs/amdgpu}"

REPO_DIR="${REPO_DIR:-/home/mgagvani/robotvision/src/camera-based-e2e}"
RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_repro_20260515_123234}"
SPLIT="${SPLIT:-val}"
BLOCKS="${BLOCKS:-3}"
TARGET_MODE="${TARGET_MODE:-composite_safety}"
TARGET_COUNT="${TARGET_COUNT:-32}"
FEATURE_BUDGET="${FEATURE_BUDGET:-64}"
TOP_ACTIVE_PER_SCENE="${TOP_ACTIVE_PER_SCENE:-8}"
MAX_SCENES_PER_FEATURE="${MAX_SCENES_PER_FEATURE:-16}"
TOP_DOWNSTREAM_FEATURES="${TOP_DOWNSTREAM_FEATURES:-8}"
ALPHAS="${ALPHAS:--1.0,1.0}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/analysis/circuit_tracing}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-${RUN_ROOT}/miopen-user-db-${SLURM_JOB_ID:-manual}}"
MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${RUN_ROOT}/miopen-cache-${SLURM_JOB_ID:-manual}}"
export MIOPEN_USER_DB_PATH
export MIOPEN_CUSTOM_CACHE_DIR

cd "${REPO_DIR}"
mkdir -p log
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

srun python analyze_sae_circuit_tracing.py \
  --run_root "${RUN_ROOT}" \
  --split "${SPLIT}" \
  --blocks "${BLOCKS}" \
  --target_mode "${TARGET_MODE}" \
  --target_count "${TARGET_COUNT}" \
  --feature_budget "${FEATURE_BUDGET}" \
  --top_active_per_scene "${TOP_ACTIVE_PER_SCENE}" \
  --max_scenes_per_feature "${MAX_SCENES_PER_FEATURE}" \
  --top_downstream_features "${TOP_DOWNSTREAM_FEATURES}" \
  --alphas="${ALPHAS}" \
  --batch_size "${BATCH_SIZE}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS}
