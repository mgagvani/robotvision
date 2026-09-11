#!/bin/bash
# Full-validation (106,360 frames) ADE for the rows of an ADE table, one row per array task,
# then aggregate:  sbatch --array=0-N scripts/run_waymo_full_val_ade.slurm
#                  MODE=aggregate sbatch scripts/run_waymo_full_val_ade.slurm
#SBATCH --job-name=ade-full-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/ade_eval/full_val}
TABLE=${TABLE:-$OUTPUT_ROOT/ade_eval/ade_table.md}
if [[ "${MODE:-row}" == "aggregate" ]]; then
  "$PYTHON_BIN" evaluate_waymo_ade_full_val_checkpoint.py --aggregate --output_dir "$OUTPUT_DIR"
else
  "$PYTHON_BIN" evaluate_waymo_ade_full_val_checkpoint.py \
    --data_dir "$DATA_DIR" \
    --table "$TABLE" \
    --row_index "$SLURM_ARRAY_TASK_ID" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "${BATCH_SIZE:-32}"
fi
