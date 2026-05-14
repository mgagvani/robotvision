#!/bin/bash
#SBATCH --job-name=sae_ade_int
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=04:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-ade-int-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${PROJECT_DIR}/camera-e2e-epoch=04-val_loss=2.90.ckpt}"
SAE_BLOCK="${SAE_BLOCK:-3}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/analysis/block_${SAE_BLOCK}/ade_intervention_all}"
FEATURES="${FEATURES:-all}"
RELEVANT_SCENES_PER_FEATURE="${RELEVANT_SCENES_PER_FEATURE:-384}"
ALPHAS="${ALPHAS:-0,0.5,1.0,2.0}"
SCENE_SELECTION="${SCENE_SELECTION:-relevant}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FEATURE_START="${FEATURE_START:-$((TASK_ID * CHUNK_SIZE))}"
FEATURE_END="${FEATURE_END:-$((FEATURE_START + CHUNK_SIZE))}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/ade_intervention_rows_${FEATURE_START}_${FEATURE_END}_${SPLIT}.csv}"

cd "${PROJECT_DIR}"

mkdir -p "${OUTPUT_DIR}"

echo "Running SAE ADE intervention analysis"
echo "RUN_ROOT=${RUN_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_CSV=${OUTPUT_CSV}"
echo "FEATURES=${FEATURES}"
echo "SAE_BLOCK=${SAE_BLOCK}"
echo "SPLIT=${SPLIT}"
echo "FEATURE_START=${FEATURE_START}"
echo "FEATURE_END=${FEATURE_END}"
echo "RELEVANT_SCENES_PER_FEATURE=${RELEVANT_SCENES_PER_FEATURE}"
echo "ALPHAS=${ALPHAS}"
echo "SCENE_SELECTION=${SCENE_SELECTION}"
echo "RANDOM_SEEDS=${RANDOM_SEEDS}"

python analyze_sae_ade_intervention.py \
    --run_root "${RUN_ROOT}" \
    --planner_checkpoint "${PLANNER_CHECKPOINT}" \
    --split "${SPLIT}" \
    --features "${FEATURES}" \
    --sae_block "${SAE_BLOCK}" \
    --feature_start "${FEATURE_START}" \
    --feature_end "${FEATURE_END}" \
    --alphas "${ALPHAS}" \
    --relevant_scenes_per_feature "${RELEVANT_SCENES_PER_FEATURE}" \
    --scene_selection "${SCENE_SELECTION}" \
    --random_seeds "${RANDOM_SEEDS}" \
    --device "${DEVICE}" \
    --output_csv "${OUTPUT_CSV}"
