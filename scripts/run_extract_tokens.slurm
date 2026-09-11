#!/bin/bash
# Planner-query tokens of every block for the train and val splits (SAE training data, analysis inputs).
#SBATCH --job-name=extract-tokens
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
SPLITS=(train val)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$RUN_ROOT/tokens"
"$PYTHON_BIN" extract_planner_tok.py \
  --checkpoint "$PLANNER_CHECKPOINT" \
  --data_dir "$DATA_DIR" \
  --index_file "index_${SPLIT}.pkl" \
  --output_path "$RUN_ROOT/tokens/planner_tokens_${SPLIT}.pt" \
  --batch_size 64 --nw 8
