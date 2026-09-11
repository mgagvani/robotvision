#!/bin/bash
# Latent steering sweep: intervene on every block-3 feature at alpha in {0, 0.5, 1, 2} over its
# 64 most-active validation scenes and score controllability (Appendix "Latent Steering").
#SBATCH --job-name=sae-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
SAE_BLOCK=${SAE_BLOCK:-3}
OUT=${OUTPUT_DIR:-$RUN_ROOT/analysis/block_$SAE_BLOCK}
"$PYTHON_BIN" analyze_sae_control.py \
  --run_root "$RUN_ROOT" \
  --planner_checkpoint "$PLANNER_CHECKPOINT" \
  --split val \
  --sae_block "$SAE_BLOCK" \
  --data_dir "$DATA_DIR" \
  --index_file index_val.pkl \
  --device cuda \
  --encode_batch_size 1024 \
  --relevant_scenes_per_feature 64 \
  --alphas 0,0.5,1.0,2.0 \
  --output_dir "$OUT"
