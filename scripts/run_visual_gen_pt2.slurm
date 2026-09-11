#!/bin/bash
# Counterfactual visual edits, stage 2: replay original vs edited frames through the frozen planner,
# encode block-3 tokens with the SAE, and write per-feature activation deltas and the feature summary
# to $VISUAL_GEN_ROOT/sae_analysis/.
#SBATCH --job-name=visual-gen-pt2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
"$PYTHON_BIN" analyze_sae_visual_gen_pt2.py \
  --manifest "$VISUAL_GEN_ROOT/manifest.jsonl" \
  --data_dir "$DATA_DIR" \
  --run_root "$RUN_ROOT" \
  --planner_checkpoint "$PLANNER_CHECKPOINT" \
  --split val \
  --sae_block "${SAE_BLOCK:-3}" \
  --batch_size 8 --encode_batch_size 4096 \
  --output_dir "$VISUAL_GEN_ROOT/sae_analysis"
