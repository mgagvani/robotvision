#!/bin/bash
#SBATCH --job-name=ipad
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=56:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --account=csso
#SBATCH --partition=a10

module load conda
# Sourcing conda.sh makes activate work in non-interactive (sbatch) shells
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate robotvision
pip install pytorch-lightning
# Use the python that activation puts on PATH.
# Some clusters don't mount the original conda env prefix on compute nodes.
PYTHON="$(command -v python || true)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "PYTHON=${PYTHON}"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: python not found after conda activate."
  echo "If this is a shared-filesystem issue, recreate the env in a path mounted on compute nodes (e.g. under \$HOME)."
  conda info --envs || true
  exit 2
fi

# Ensure pytorch_lightning is available (required by train.py).
if ! "$PYTHON" -c "import pytorch_lightning as pl; print('pytorch_lightning', pl.__version__)" 2>/dev/null; then
  echo "ERROR: pytorch_lightning not found in env: ${CONDA_PREFIX:-unknown}"
  echo "Install it in the same env you use for this job, e.g.:"
  echo "  conda activate robotvision"
  echo "  pip install pytorch-lightning"
  exit 3
fi
# Waymo E2E data; index_train.pkl and index_val.pkl must be in PROJECT_DIR
DATA_DIR="/scratch/gilbreth/mathur91/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0"
PROJECT_DIR="/home/mathur91/robotvision/src/camera-based-e2e"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Scorer targets: rfs = per-proposal RFS quality at 3s/5s (× jerk comfort). Override with EXTRA_ARGS, e.g. --score_target_type l1
"$PYTHON" train.py \
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
  --rfs_weight 0.2 \
  --diversity_weight 0.0 \
  --score_weight 1.0 \
  --score_warmup_epochs 2 \
  --score_temperature 5.0 \
  --score_loss_type bce \
  --score_target_type rfs \
  --score_rank_weight 0.2 \
  --score_margin 0.2 \
  --score_topk 0 \
  --comfort_jerk_threshold 5.0 \
  --prev_weight 0.1 \
  --grad_clip 1.0 \
  --log_every_n_steps 100 \
  ${EXTRA_ARGS:-}

# Auto-refresh cross-run scorer experiment ranking plots/report.
LOGS_ROOT="$(dirname "$DATA_DIR")/logs"
if [ -d "$LOGS_ROOT" ]; then
  "$PYTHON" score_experiment_viz.py --logs_root "$LOGS_ROOT" --latest_n 24 || true
fi