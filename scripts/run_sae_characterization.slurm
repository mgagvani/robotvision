#!/bin/bash
# Block-3 latent characterization on the validation split: intent correlation and
# feature co-activation clustering at the paper's three thresholds (0.35 is the primary one).
#SBATCH --job-name=sae-characterization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
SAE_BLOCK=${SAE_BLOCK:-3}
OUT=${OUTPUT_DIR:-$RUN_ROOT/analysis/block_$SAE_BLOCK}
"$PYTHON_BIN" analyze_sae_intent.py --run_root "$RUN_ROOT" --split val --sae_block "$SAE_BLOCK" --output_dir "$OUT"
for TAU in 0.20 0.35 0.40; do
  "$PYTHON_BIN" analyze_sae_feature_correlations.py --run_root "$RUN_ROOT" --split val --sae_block "$SAE_BLOCK" \
    --min_active_frac 0.005 --max_active_frac 0.50 --corr_threshold "$TAU" --min_cluster_size 3 \
    --output_dir "$OUT/correlations_tau$TAU"
done
