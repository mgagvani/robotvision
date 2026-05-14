#!/bin/bash
#SBATCH --job-name=sae_pipeline
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-gpu=10
#SBATCH --time=01:00:00
#SBATCH --output=/home/mgagvani/robotvision/src/camera-based-e2e/slurm-sae-%j.out

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amdgpu

PROJECT_DIR="/home/mgagvani/robotvision/src/camera-based-e2e"
SBATCH_DIR="/home/mgagvani/robotvision/sbatch"
cd "${PROJECT_DIR}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME="${HF_HOME:-/scratch/negishi/mgagvani/hfcache}"
export WAYMO_DATA_DIR="${WAYMO_DATA_DIR:-/scratch/negishi/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/camera-e2e-epoch=04-val_loss=2.90.ckpt}"
RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"
TOKENS_DIR="${TOKENS_DIR:-${RUN_ROOT}/tokens}"
SAE_OUTPUT_DIR="${SAE_OUTPUT_DIR:-${RUN_ROOT}/model}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_ROOT}/analysis}"

EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
N_WORKERS="${N_WORKERS:-0}"

mkdir -p "${TOKENS_DIR}" "${SAE_OUTPUT_DIR}" "${ANALYSIS_DIR}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

TRAIN_TOKEN_PATH="${TOKENS_DIR}/planner_tokens_train.pt"
VAL_TOKEN_PATH="${TOKENS_DIR}/planner_tokens_val.pt"
BLOCKS_RAW="${BLOCKS:-0,1,2,3}"
BLOCKS="${BLOCKS_RAW//,/ }"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4096}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
K="${K:-auto}"
K_AUX="${K_AUX:-512}"
LAMBDA_AUX="${LAMBDA_AUX:-0.03125}"
DEAD_STEPS_THRESHOLD="${DEAD_STEPS_THRESHOLD:-500}"
GEOMETRIC_MEDIAN_SAMPLES="${GEOMETRIC_MEDIAN_SAMPLES:-10000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SPLIT="${SPLIT:-val}"
CONTROL_RELEVANT_SCENES="${CONTROL_RELEVANT_SCENES:-64}"
CONTROL_ALPHAS="${CONTROL_ALPHAS:-0,0.5,1.0,2.0}"
GATED_ALPHAS="${GATED_ALPHAS:-1.0,2.0}"
GATED_THRESHOLD_QUANTILES="${GATED_THRESHOLD_QUANTILES:-0.5,0.75,0.9,0.95}"
GATED_PROXY_METRIC_QUANTILES="${GATED_PROXY_METRIC_QUANTILES:-0.75,0.9}"
GATED_MIN_PROXY_ACCEPT_COUNT="${GATED_MIN_PROXY_ACCEPT_COUNT:-32}"
GATED_BATCH_SIZE="${GATED_BATCH_SIZE:-2048}"
GATED_CHUNK_SIZE="${GATED_CHUNK_SIZE:-128}"
ADE_RELEVANT_SCENES="${ADE_RELEVANT_SCENES:-384}"
ADE_ALPHAS="${ADE_ALPHAS:-0,0.5,1.0,2.0}"
ADE_SCENE_SELECTION="${ADE_SCENE_SELECTION:-relevant}"
ADE_RANDOM_SEEDS="${ADE_RANDOM_SEEDS:-0}"
ADE_CHUNK_SIZE="${ADE_CHUNK_SIZE:-128}"

echo "Running SAE pipeline from ${PROJECT_DIR}"
echo "WAYMO_DATA_DIR=${WAYMO_DATA_DIR}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "BLOCKS=${BLOCKS}"
echo "DEVICE=${DEVICE}"

submit_dependency() {
    local script_path="$1"
    local dependency="$2"
    shift 2
    if [[ -n "${dependency}" ]]; then
        sbatch --parsable --dependency="${dependency}" "$@" "${script_path}"
    else
        sbatch --parsable "$@" "${script_path}"
    fi
}

latent_dim_from_model() {
    python - <<'PY'
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel

model = DeepMonocularModel(
    feature_extractor=SAMFeatures(
        model_name="timm/vit_pe_spatial_small_patch16_512.fb",
        frozen=True,
    ),
    out_dim=20 * 2,
    n_blocks=4,
    n_proposals=50,
)
print(int(model.feature_dim))
PY
}

latent_dim="${LATENT_DIM:-$(latent_dim_from_model)}"
gated_chunks=$(( (latent_dim + GATED_CHUNK_SIZE - 1) / GATED_CHUNK_SIZE ))
ade_chunks=$(( (latent_dim + ADE_CHUNK_SIZE - 1) / ADE_CHUNK_SIZE ))

extract_export="ALL,RUN_ROOT=${RUN_ROOT},TOKENS_DIR=${TOKENS_DIR},CHECKPOINT_PATH=${CHECKPOINT_PATH},DEVICE=${DEVICE},EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE},N_WORKERS=${N_WORKERS}"
if [[ -n "${N_ITEMS:-}" ]]; then
    extract_export+=",N_ITEMS=${N_ITEMS}"
fi

extract_train_job=$(submit_dependency "${SBATCH_DIR}/extract_tok.sbatch" "" \
    --export="${extract_export},SPLIT=train")
extract_val_job=$(submit_dependency "${SBATCH_DIR}/extract_tok.sbatch" "" \
    --export="${extract_export},SPLIT=val")

extract_dependency="afterok:${extract_train_job}:${extract_val_job}"

declare -a final_job_ids=()

for sae_block in ${BLOCKS}; do
    train_job=$(submit_dependency "${SBATCH_DIR}/train_SAE.sbatch" "${extract_dependency}" \
        --export=ALL,RUN_ROOT="${RUN_ROOT}",TOKENS_DIR="${TOKENS_DIR}",SAE_OUTPUT_DIR="${SAE_OUTPUT_DIR}",TRAIN_TOKEN_PATH="${TRAIN_TOKEN_PATH}",VAL_TOKEN_PATH="${VAL_TOKEN_PATH}",SAE_BLOCK="${sae_block}",TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",LR="${LR}",MAX_EPOCHS="${MAX_EPOCHS}",K="${K}",K_AUX="${K_AUX}",LAMBDA_AUX="${LAMBDA_AUX}",DEAD_STEPS_THRESHOLD="${DEAD_STEPS_THRESHOLD}",GEOMETRIC_MEDIAN_SAMPLES="${GEOMETRIC_MEDIAN_SAMPLES}",GRAD_CLIP="${GRAD_CLIP}",DEVICE="${DEVICE}")
    train_dependency="afterok:${train_job}"

    analysis_job=$(submit_dependency "${SBATCH_DIR}/sae_analysis_block.sbatch" "${train_dependency}" \
        --export=ALL,RUN_ROOT="${RUN_ROOT}",PLANNER_CHECKPOINT="${CHECKPOINT_PATH}",SPLIT="${SPLIT}",SAE_BLOCK="${sae_block}",DEVICE="${DEVICE}",ENCODE_BATCH_SIZE="${TRAIN_BATCH_SIZE}",RELEVANT_SCENES_PER_FEATURE="${CONTROL_RELEVANT_SCENES}",ALPHAS="${CONTROL_ALPHAS}")
    final_job_ids+=("${analysis_job}")

    gated_job=$(submit_dependency "${SBATCH_DIR}/sae_gated_ade.sbatch" "${train_dependency}" \
        --array="0-$((gated_chunks - 1))" \
        --export=ALL,RUN_ROOT="${RUN_ROOT}",PLANNER_CHECKPOINT="${CHECKPOINT_PATH}",SPLIT="${SPLIT}",SAE_BLOCK="${sae_block}",FEATURES=all,ALPHAS="${GATED_ALPHAS}",THRESHOLD_QUANTILES="${GATED_THRESHOLD_QUANTILES}",PROXY_METRIC_QUANTILES="${GATED_PROXY_METRIC_QUANTILES}",MIN_PROXY_ACCEPT_COUNT="${GATED_MIN_PROXY_ACCEPT_COUNT}",BATCH_SIZE="${GATED_BATCH_SIZE}",CHUNK_SIZE="${GATED_CHUNK_SIZE}",DEVICE="${DEVICE}")
    final_job_ids+=("${gated_job}")

    ade_job=$(submit_dependency "${SBATCH_DIR}/sae_ade_intervention.sbatch" "${train_dependency}" \
        --array="0-$((ade_chunks - 1))" \
        --export=ALL,RUN_ROOT="${RUN_ROOT}",PLANNER_CHECKPOINT="${CHECKPOINT_PATH}",SPLIT="${SPLIT}",SAE_BLOCK="${sae_block}",FEATURES=all,RELEVANT_SCENES_PER_FEATURE="${ADE_RELEVANT_SCENES}",ALPHAS="${ADE_ALPHAS}",SCENE_SELECTION="${ADE_SCENE_SELECTION}",RANDOM_SEEDS="${ADE_RANDOM_SEEDS}",CHUNK_SIZE="${ADE_CHUNK_SIZE}",DEVICE="${DEVICE}")
    final_job_ids+=("${ade_job}")

    echo "Submitted block ${sae_block}: train=${train_job} analysis=${analysis_job} gated=${gated_job} ade=${ade_job}"
done

merge_dependency="afterok:$(IFS=:; echo "${final_job_ids[*]}")"
merge_job=$(submit_dependency "${SBATCH_DIR}/sae_merge_analysis.sbatch" "${merge_dependency}" \
    --export=ALL,RUN_ROOT="${RUN_ROOT}",ANALYSIS_DIR="${ANALYSIS_DIR}")

echo "Submitted token jobs: train=${extract_train_job} val=${extract_val_job}"
echo "Submitted merge job: ${merge_job}"
echo "SAE pipeline submission complete."

# Example overrides:
# N_ITEMS=2048 MAX_EPOCHS=5 sbatch sbatch/sae_pipeline.sbatch
