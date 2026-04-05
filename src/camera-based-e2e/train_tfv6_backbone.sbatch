#!/bin/bash
#SBATCH --job-name=tfv6-backbone
#SBATCH --output=/scratch/gilbreth/%u/robotvision/src/camera-based-e2e/slurm_logs/tfv6-backbone-%j.out
#SBATCH --error=/scratch/gilbreth/%u/robotvision/src/camera-based-e2e/slurm_logs/tfv6-backbone-%j.err
#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=84:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -eo pipefail

module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shar1159/conda_envs/lead_ltf
set -u

if [ -d "/depot/mlp/data/$USER/robotvision" ]; then
    REPO_ROOT="/depot/mlp/data/$USER/robotvision"
elif [ -d "$HOME/robotvision" ]; then
    REPO_ROOT="$HOME/robotvision"
elif [ -d "/scratch/gilbreth/$USER/robotvision" ]; then
    REPO_ROOT="/scratch/gilbreth/$USER/robotvision"
else
    echo "Could not find robotvision repo under /depot/mlp/data/$USER, $HOME, or /scratch/gilbreth/$USER"
    exit 1
fi

PROJECT_ROOT="$REPO_ROOT/src/camera-based-e2e"
cd "$PROJECT_ROOT" || exit 1

DATA_DIR_DEFAULT="/scratch/gilbreth/$USER/waymo_data/waymo_open_dataset_end_to_end_camera_v_1_0_0"
if [ ! -d "$DATA_DIR_DEFAULT" ]; then
    DATA_DIR_DEFAULT="/scratch/gilbreth/chang899/waymo_data/waymo_open_dataset_end_to_end_camera_v_1_0_0"
fi
if [ ! -d "$DATA_DIR_DEFAULT" ]; then
    DATA_DIR_DEFAULT="/scratch/gilbreth/$USER/waymo_preprocessed"
fi
if [ ! -d "$DATA_DIR_DEFAULT" ]; then
    DATA_DIR_DEFAULT="/scratch/gilbreth/chang899/waymo_preprocessed"
fi

DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"
TRAIN_INDEX="${TRAIN_INDEX:-index_train.pkl}"
VAL_INDEX="${VAL_INDEX:-index_val.pkl}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-10}"
N_TRAIN_ITEMS="${N_TRAIN_ITEMS:-250000}"
N_VAL_ITEMS="${N_VAL_ITEMS:-25000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RFS_WEIGHT="${RFS_WEIGHT:-0.0}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/gilbreth/$USER/tfv6_backbone_runs}"
RESUME_CKPT="${RESUME_CKPT:-}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
DISABLE_PROGRESS_BAR="${DISABLE_PROGRESS_BAR:-1}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/slurm_logs"

export PYTHONUNBUFFERED=1

python -c "import sys; print('python_executable=', sys.executable)"
python -c "import torch, pytorch_lightning, timm; print('torch=', torch.__version__, 'pl=', pytorch_lightning.__version__, 'timm=', timm.__version__)"

if [ ! -d "$DATA_DIR" ]; then
    echo "DATA_DIR does not exist: $DATA_DIR"
    exit 1
fi

resolve_index() {
    local idx="$1"
    if [[ "$idx" = /* ]] && [ -f "$idx" ]; then
        echo "$idx"
        return 0
    fi
    if [ -f "$idx" ]; then
        echo "$idx"
        return 0
    fi
    if [ -f "$PROJECT_ROOT/$idx" ]; then
        echo "$PROJECT_ROOT/$idx"
        return 0
    fi
    if [ -f "$DATA_DIR/$idx" ]; then
        echo "$DATA_DIR/$idx"
        return 0
    fi
    return 1
}

if ! TRAIN_INDEX_RESOLVED="$(resolve_index "$TRAIN_INDEX")"; then
    echo "Could not resolve TRAIN_INDEX: $TRAIN_INDEX"
    echo "Checked absolute path, current dir, $PROJECT_ROOT, and $DATA_DIR"
    exit 1
fi
if ! VAL_INDEX_RESOLVED="$(resolve_index "$VAL_INDEX")"; then
    echo "Could not resolve VAL_INDEX: $VAL_INDEX"
    echo "Checked absolute path, current dir, $PROJECT_ROOT, and $DATA_DIR"
    exit 1
fi

echo "Using DATA_DIR=$DATA_DIR"
echo "Using TRAIN_INDEX=$TRAIN_INDEX_RESOLVED VAL_INDEX=$VAL_INDEX_RESOLVED"
echo "Using BATCH_SIZE=$BATCH_SIZE LR=$LR EPOCHS=$EPOCHS"
echo "Using N_TRAIN_ITEMS=$N_TRAIN_ITEMS N_VAL_ITEMS=$N_VAL_ITEMS NUM_WORKERS=$NUM_WORKERS PREFETCH_FACTOR=$PREFETCH_FACTOR"
echo "Using LOG_EVERY_N_STEPS=$LOG_EVERY_N_STEPS CHECK_VAL_EVERY_N_EPOCH=$CHECK_VAL_EVERY_N_EPOCH DISABLE_PROGRESS_BAR=$DISABLE_PROGRESS_BAR"
echo "Using OUTPUT_DIR=$OUTPUT_DIR"
if [ -n "$RESUME_CKPT" ]; then
    echo "Using RESUME_CKPT=$RESUME_CKPT"
fi

CMD=(
  python train_tfv6_backbone.py
  --data_dir "$DATA_DIR"
  --train_index "$TRAIN_INDEX_RESOLVED"
  --val_index "$VAL_INDEX_RESOLVED"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --max_epochs "$EPOCHS"
  --n_train_items "$N_TRAIN_ITEMS"
  --n_val_items "$N_VAL_ITEMS"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --log_every_n_steps "$LOG_EVERY_N_STEPS"
  --check_val_every_n_epoch "$CHECK_VAL_EVERY_N_EPOCH"
  --rfs_weight "$RFS_WEIGHT"
  --output_dir "$OUTPUT_DIR"
)

if [ "${FREEZE_BACKBONE:-0}" = "1" ]; then
    CMD+=(--freeze_backbone)
fi
if [ "${COMPILE:-0}" = "1" ]; then
    CMD+=(--compile)
fi
if [ "${PROFILE:-0}" = "1" ]; then
    CMD+=(--profile)
fi
if [ -n "$RESUME_CKPT" ]; then
    CMD+=(--resume_ckpt "$RESUME_CKPT")
fi
if [ "${DISABLE_PROGRESS_BAR:-0}" = "1" ]; then
    CMD+=(--disable_progress_bar)
fi

srun "${CMD[@]}"