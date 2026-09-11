#!/bin/bash
# Train one planner on WOD-E2E (paper: SMTS on 250k samples, 10 epochs, global batch 128, lr 1e-4).
#   MODEL=deepmonocular|gtrs|drivor   TRAIN_ITEMS=250000   sbatch scripts/run_train_waymo.slurm
#SBATCH --job-name=train-waymo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=36:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
MODEL=${MODEL:-deepmonocular}
TRAIN_ITEMS=${TRAIN_ITEMS:-250000}
VAL_ITEMS=${VAL_ITEMS:-25000}
LR=${LR:-1e-4}
MAX_EPOCHS=${MAX_EPOCHS:-10}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
BATCH_SIZE=$((GLOBAL_BATCH_SIZE / GPUS_PER_NODE))

echo "model=$MODEL train_items=$TRAIN_ITEMS val_items=$VAL_ITEMS global_batch=$GLOBAL_BATCH_SIZE gpus=$GPUS_PER_NODE"
srun --ntasks="$GPUS_PER_NODE" --ntasks-per-node="$GPUS_PER_NODE" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1 \
  "$PYTHON_BIN" train.py \
    --data_dir "$DATA_DIR" \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --train_items "$TRAIN_ITEMS" \
    --val_items "$VAL_ITEMS"
