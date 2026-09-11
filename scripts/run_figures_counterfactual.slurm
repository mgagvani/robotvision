#!/bin/bash
# Counterfactual-edit figures from the stage-2 outputs: Fig. 3 (main paper), the expected-direction
# gallery, and the supplement example/failure panels. Figures land in $FIG_DIR.
#SBATCH --job-name=figures-counterfactual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-$USER}
SAE_BLOCK=${SAE_BLOCK:-3}
FIG_DIR=${FIG_DIR:-$OUTPUT_ROOT/figures}
MANIFEST=$VISUAL_GEN_ROOT/manifest.jsonl
PAIR_CSV=$VISUAL_GEN_ROOT/sae_analysis/sae_visual_gen_pairs_block_$SAE_BLOCK.csv
SUMMARY_CSV=$VISUAL_GEN_ROOT/sae_analysis/sae_visual_gen_feature_summary_block_$SAE_BLOCK.csv
TOKEN_PATH=$RUN_ROOT/tokens/planner_tokens_val.pt
mkdir -p "$FIG_DIR"
COMMON=(--manifest "$MANIFEST" --pair_csv "$PAIR_CSV" --data_dir "$DATA_DIR" --index_file index_val.pkl
        --token_path "$TOKEN_PATH" --planner_checkpoint "$PLANNER_CHECKPOINT")

# Fig. 3
"$PYTHON_BIN" visualizations/paper/extract_counterfactual_trajectories.py "${COMMON[@]}" \
  --output "$FIG_DIR/fig3_counterfactual_trajectories.json"
"$PYTHON_BIN" visualizations/paper/plot_counterfactual_figure.py \
  --manifest "$MANIFEST" --analysis_dir "$VISUAL_GEN_ROOT/sae_analysis" --data_dir "$DATA_DIR" --index_file index_val.pkl \
  --sae_block "$SAE_BLOCK" --trajectory_cache "$FIG_DIR/fig3_counterfactual_trajectories.json" \
  --output "$FIG_DIR/fig3_counterfactual.png" --pdf_output "$FIG_DIR/fig3_counterfactual.pdf"

# Expected-direction gallery (also the source of the supplement examples)
"$PYTHON_BIN" visualizations/paper/extract_counterfactual_trajectories.py "${COMMON[@]}" \
  --directions green_to_red red_to_green add_stop_sign add_pedestrian remove_stop_sign remove_pedestrian \
  --per_direction 5 --candidate_pool 40 --expected_forward_delta \
  --exclude_cache "$FIG_DIR/fig3_counterfactual_trajectories.json" \
  --output "$FIG_DIR/counterfactual_gallery_expected_trajectories.json"
"$PYTHON_BIN" visualizations/paper/plot_counterfactual_gallery.py \
  --trajectory_cache "$FIG_DIR/counterfactual_gallery_expected_trajectories.json" --data_dir "$DATA_DIR" --index_file index_val.pkl \
  --output "$FIG_DIR/counterfactual_gallery_expected.png" --pdf_output "$FIG_DIR/counterfactual_gallery_expected.pdf"

# Supplement: additional examples and failure cases
"$PYTHON_BIN" visualizations/paper/plot_supp_edit_examples.py \
  --trajectory_cache "$FIG_DIR/counterfactual_gallery_expected_trajectories.json" \
  --failures_trajectory_cache "$FIG_DIR/supp_edit_failures_trajectories.json" \
  --summary_csv "$SUMMARY_CSV" --pairs_csv "$PAIR_CSV" --data_dir "$DATA_DIR" --index_file index_val.pkl \
  --examples_output "$FIG_DIR/supp_edit_examples.png" --failures_output "$FIG_DIR/supp_edit_failures.png"
