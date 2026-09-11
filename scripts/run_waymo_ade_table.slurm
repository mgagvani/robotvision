#!/bin/bash
# Open-loop ADE@3s / ADE@5s for every checkpoint in $CHECKPOINT_DIR on a validation subset.
#SBATCH --job-name=ade-table
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/ade_eval}
"$PYTHON_BIN" evaluate_waymo_ade_table.py \
  --data_dir "$DATA_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "${BATCH_SIZE:-32}"
