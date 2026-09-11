#!/bin/bash
# WOD-E2E test-set submission tarball for one checkpoint (Rater Feedback Score is computed by Waymo's server).
#   MODEL=deepmonocular CHECKPOINT=/path/to.ckpt sbatch scripts/run_waymo_submission.slurm
#SBATCH --job-name=wod-submission
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
MODEL=${MODEL:-deepmonocular}
CHECKPOINT=${CHECKPOINT:-$PLANNER_CHECKPOINT}
NAME=${NAME:-${MODEL}_$(basename "$CHECKPOINT" .ckpt)}
SUBMISSION_ROOT=${SUBMISSION_ROOT:-$OUTPUT_ROOT/submissions}
mkdir -p "$SUBMISSION_ROOT"
"$PYTHON_BIN" submission.py \
  --data_dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --model_type "$MODEL" \
  --output_file "$SUBMISSION_ROOT/$NAME" \
  --tarball "$SUBMISSION_ROOT/$NAME.tar.gz" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --unique_method_name "$NAME"
