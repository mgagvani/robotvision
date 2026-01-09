#!/bin/bash
#SBATCH --job-name=ablation_exp
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --array=1-30
#SBATCH --output=./outputs/slurm_logs/slurm-%A_%a.out
#SBATCH --error=./outputs/slurm_logs/slurm-%A_%a.err

# Load modules
module load conda

# Activate conda environment
source ~/.bashrc 2>/dev/null || true
conda activate waymo_dataloader || source activate waymo_dataloader

# Change to experiment directory
EXPERIMENT_DIR="/home/mathur91/robotvision/src/camera-based-e2e/experiments/ablation"
cd "$EXPERIMENT_DIR"

# Path to data directory
DATA_DIR="/scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0/"

# Path to index files (relative to camera-based-e2e directory)
INDEX_DIR="/home/mathur91/robotvision/src/camera-based-e2e"

# Load experiment configuration
CONFIG_FILE="$EXPERIMENT_DIR/experiment_configs.json"

# Get experiment config for this array task (all experiments)
EXP_CONFIG=$(python3 -c "
import json
import sys
with open('$CONFIG_FILE', 'r') as f:
    configs = json.load(f)
idx = int(sys.argv[1]) - 1  # Array task ID is 1-30, convert to 0-29
if idx >= len(configs):
    print('ERROR: Array task ID out of range', file=sys.stderr)
    sys.exit(1)
config = configs[idx]
print(json.dumps(config))
" $SLURM_ARRAY_TASK_ID)

# Extract parameters
EXP_NAME=$(echo "$EXP_CONFIG" | python3 -c "import json, sys; print(json.load(sys.stdin)['name'])")
MODEL_TYPE=$(echo "$EXP_CONFIG" | python3 -c "import json, sys; print(json.load(sys.stdin)['model_type'])")
USE_POS=$(echo "$EXP_CONFIG" | python3 -c "import json, sys; print('--use_position' if json.load(sys.stdin)['use_position'] else '')")
USE_VEL=$(echo "$EXP_CONFIG" | python3 -c "import json, sys; print('--use_velocity' if json.load(sys.stdin)['use_velocity'] else '')")
USE_ACC=$(echo "$EXP_CONFIG" | python3 -c "import json, sys; print('--use_acceleration' if json.load(sys.stdin)['use_acceleration'] else '')")
USE_INTENT=$(echo "$EXP_CONFIG" | python3 -c "import json, sys; print('--use_intent' if json.load(sys.stdin)['use_intent'] else '')")

# Build command
CMD="python train_ablation.py \
    --data_dir $DATA_DIR \
    --batch_size 16 \
    --lr 0.0001 \
    --max_epochs 15 \
    --model_type $MODEL_TYPE \
    --experiment_name $EXP_NAME \
    --output_dir ./outputs \
    $USE_POS $USE_VEL $USE_ACC $USE_INTENT"

# Set PYTHONPATH to include parent directory for imports
export PYTHONPATH="/home/mathur91/robotvision/src/camera-based-e2e:$PYTHONPATH"

# Create slurm_logs directory if it doesn't exist
mkdir -p "$EXPERIMENT_DIR/outputs/slurm_logs"

# Copy index files to experiment directory if they don't exist
if [ ! -f "$EXPERIMENT_DIR/index_train.pkl" ]; then
    cp "$INDEX_DIR/index_train.pkl" "$EXPERIMENT_DIR/"
fi
if [ ! -f "$EXPERIMENT_DIR/index_val.pkl" ]; then
    cp "$INDEX_DIR/index_val.pkl" "$EXPERIMENT_DIR/"
fi

echo "=========================================="
echo "Starting experiment: $EXP_NAME"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Command: $CMD"
echo "Time: $(date)"
echo "=========================================="

# Run training
eval $CMD

EXIT_CODE=$?

echo "=========================================="
echo "Experiment $EXP_NAME completed"
echo "Exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="

exit $EXIT_CODE


