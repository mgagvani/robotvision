#!/bin/bash
# Run proposal training on the current machine (e.g. login) with visible GPU.
# Same args as run_train_proposal.sbatch, without Slurm. Does not pip-install PL.

set -euo pipefail

module load conda 2>/dev/null || true
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate robotvision

DATA_DIR="/scratch/gilbreth/mathur91/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0"
PROJECT_DIR="/home/mathur91/robotvision/src/camera-based-e2e"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

exec python train.py \
  --data_dir "$DATA_DIR" \
  --model_type proposal \
  --backbone resnet \
  --no_wandb \
  --batch_size 16 \
  --lr 1e-4 \
  --max_epochs 20 \
  --num_proposals 16 \
  --num_refinement_steps 4 \
  --smoothness_weight 0.0 \
  --comfort_weight 0.0 \
  --diversity_weight 0.0 \
  --score_weight 1.0 \
  --score_warmup_epochs 2 \
  --score_temperature 5.0 \
  --score_loss_type bce \
  --score_target_type l1 \
  --score_rank_weight 0.2 \
  --score_margin 0.2 \
  --score_topk 0 \
  --comfort_jerk_threshold 5.0 \
  --prev_weight 0.1 \
  --grad_clip 1.0 \
  --log_every_n_steps 100 \
  ${EXTRA_ARGS:-}
