#!/bin/bash
# Train one TopK-AuxK SAE per planner block on the extracted tokens (paper: 384->384, k=24,
# k_aux=512, alpha_aux=1/32, dead after 500 steps, 100 epochs, batch 4096, lr 1e-4, seed 42).
# Writes $RUN_ROOT/model/block_{0..3}/sae_checkpoint.pt (the shipped sae_checkpoints/ are these files).
#SBATCH --job-name=train-sae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
"$PYTHON_BIN" train_sae.py \
  --output_dir "$RUN_ROOT" \
  --train_dataset "$RUN_ROOT/tokens/planner_tokens_train.pt" \
  --val_dataset "$RUN_ROOT/tokens/planner_tokens_val.pt" \
  --blocks "${BLOCKS:-all}" \
  --k 24 --k_aux 512 --lambda_aux 0.03125 --dead_steps_threshold 500 \
  --batch_size 4096 --lr 1e-4 --max_epochs 100 --grad_clip 1.0 --geometric_median_samples 10000 --seed 42
