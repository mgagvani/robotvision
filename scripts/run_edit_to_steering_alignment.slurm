#!/bin/bash
# Fig. 5b: do features moved by counterfactual edits steer the planner in the implied direction?
# Needs run_visual_gen_pt2 (feature summary) and run_sae_control (stat rows).
#SBATCH --job-name=edit-steer-align
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=01:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
SAE_BLOCK=${SAE_BLOCK:-3}
ANALYSIS_DIR=${ANALYSIS_DIR:-$RUN_ROOT/analysis/block_$SAE_BLOCK}
OUT=${OUTPUT_DIR:-$ANALYSIS_DIR/edit_sensitivity}
mkdir -p "$OUT"
"$PYTHON_BIN" visualizations/paper/edit_sensitivity/plot_edit_to_steering_alignment.py \
  --visual_summary "$VISUAL_GEN_ROOT/sae_analysis/sae_visual_gen_feature_summary_block_$SAE_BLOCK.csv" \
  --control_stat_rows "$ANALYSIS_DIR/sae_control_stat_rows_block_${SAE_BLOCK}_val.csv" \
  --output_prefix "$OUT/fig5b_edit_to_steering_alignment" \
  --directions green_to_red,red_to_green,add_stop_sign,add_pedestrian \
  --top_k_per_direction 80
