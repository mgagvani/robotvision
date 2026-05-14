#!/bin/bash
#SBATCH --job-name=sae_gated_ade
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=04:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-gated-ade-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${PROJECT_DIR}/camera-e2e-epoch=04-val_loss=2.90.ckpt}"
SAE_BLOCK="${SAE_BLOCK:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/analysis/block_${SAE_BLOCK}/gated_ade_all}"
FEATURES="${FEATURES:-all}"
ALPHAS="${ALPHAS:-1.0,2.0}"
THRESHOLD_QUANTILES="${THRESHOLD_QUANTILES:-0.5,0.75,0.9,0.95}"
PROXY_METRIC_QUANTILES="${PROXY_METRIC_QUANTILES:-0.75,0.9}"
MIN_PROXY_ACCEPT_COUNT="${MIN_PROXY_ACCEPT_COUNT:-32}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FEATURE_START="${FEATURE_START:-$((TASK_ID * CHUNK_SIZE))}"
FEATURE_END="${FEATURE_END:-$((FEATURE_START + CHUNK_SIZE))}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/gated_rows_${FEATURE_START}_${FEATURE_END}.csv}"
OUTPUT_BEST_CSV="${OUTPUT_BEST_CSV:-${OUTPUT_DIR}/gated_best_${FEATURE_START}_${FEATURE_END}.csv}"
OUTPUT_PROXY_CSV="${OUTPUT_PROXY_CSV:-${OUTPUT_DIR}/gated_proxy_${FEATURE_START}_${FEATURE_END}.csv}"

cd "${PROJECT_DIR}"

echo "Running SAE gated ADE analysis"
echo "RUN_ROOT=${RUN_ROOT}"
echo "PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_CSV=${OUTPUT_CSV}"
echo "OUTPUT_BEST_CSV=${OUTPUT_BEST_CSV}"
echo "OUTPUT_PROXY_CSV=${OUTPUT_PROXY_CSV}"
echo "FEATURES=${FEATURES}"
echo "SAE_BLOCK=${SAE_BLOCK}"
echo "FEATURE_START=${FEATURE_START}"
echo "FEATURE_END=${FEATURE_END}"
echo "ALPHAS=${ALPHAS}"
echo "THRESHOLD_QUANTILES=${THRESHOLD_QUANTILES}"
echo "PROXY_METRIC_QUANTILES=${PROXY_METRIC_QUANTILES}"
echo "MIN_PROXY_ACCEPT_COUNT=${MIN_PROXY_ACCEPT_COUNT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "DEVICE=${DEVICE}"
echo "SPLIT=${SPLIT}"

CMD=(
    python -u analyze_sae_gated_ade.py
    --run_root "${RUN_ROOT}"
    --planner_checkpoint "${PLANNER_CHECKPOINT}"
    --split "${SPLIT}"
    --features "${FEATURES}"
    --sae_block "${SAE_BLOCK}"
    --feature_start "${FEATURE_START}"
    --feature_end "${FEATURE_END}"
    --alphas "${ALPHAS}"
    --threshold_quantiles "${THRESHOLD_QUANTILES}"
    --proxy_metric_quantiles "${PROXY_METRIC_QUANTILES}"
    --min_proxy_accept_count "${MIN_PROXY_ACCEPT_COUNT}"
    --batch_size "${BATCH_SIZE}"
    --device "${DEVICE}"
    --output_csv "${OUTPUT_CSV}"
    --output_best_csv "${OUTPUT_BEST_CSV}"
)

if [[ -n "${OUTPUT_PROXY_CSV}" ]]; then
    CMD+=(--output_proxy_csv "${OUTPUT_PROXY_CSV}")
fi

"${CMD[@]}"
