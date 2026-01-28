#!/bin/bash
#SBATCH --job-name=vlm_waypoint
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=a10
#SBATCH --account=csso
#SBATCH --mem=64G

# VLM Waypoint Training Script
# This runs the full 10-epoch training with the frozen VLM model

echo "========================================"
echo "VLM Waypoint Prediction Training"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Load conda and activate environment
module load conda
conda activate waymo_dataloader

# Set protobuf workaround
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Data directory
DATA_DIR="/scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0"

# Training parameters
BATCH_SIZE=16
LEARNING_RATE=1e-4
MAX_EPOCHS=10

echo "Configuration:"
echo "  Data dir: $DATA_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS"
echo ""

# Run training
python train.py \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_epochs $MAX_EPOCHS

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training successful!"
    echo ""
    echo "Results saved to:"
    echo "  - Checkpoints: /scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/checkpoints/"
    echo "  - Logs: /scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/logs/"
    echo "  - Visualization: ./visualizations/loss.png"
else
    echo "✗ Training failed!"
fi

exit $EXIT_CODE
