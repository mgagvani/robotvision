#!/bin/bash
# Training-set-size sweep for the data-efficiency figure: one array task per size.
#   sbatch scripts/run_train_waymo_sweep.slurm
#SBATCH --job-name=train-waymo-sweep
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gpus-per-node=4
#SBATCH --time=36:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
TRAIN_ITEM_OPTIONS=(50000 100000 150000 200000 250000)
TRAIN_ITEMS=${TRAIN_ITEM_OPTIONS[$SLURM_ARRAY_TASK_ID]}
VAL_ITEMS=${VAL_ITEMS:-$((TRAIN_ITEMS / 10))}
LR=${LR:-1e-4}
MAX_EPOCHS=${MAX_EPOCHS:-10}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
BATCH_SIZE=$((GLOBAL_BATCH_SIZE / GPUS_PER_NODE))

echo "train_items=$TRAIN_ITEMS val_items=$VAL_ITEMS global_batch=$GLOBAL_BATCH_SIZE gpus=$GPUS_PER_NODE"
srun --ntasks="$GPUS_PER_NODE" --ntasks-per-node="$GPUS_PER_NODE" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --kill-on-bad-exit=1 \
  "$PYTHON_BIN" train.py \
    --data_dir "$DATA_DIR" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --train_items "$TRAIN_ITEMS" \
    --val_items "$VAL_ITEMS"
