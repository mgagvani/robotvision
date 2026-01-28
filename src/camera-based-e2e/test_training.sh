#!/bin/bash
# Quick test to verify VLM model integrates with training pipeline

# Load conda
module load conda
conda activate waymo_dataloader

# Set protobuf workaround
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Data directory (actual data is in subdirectory)
DATA_DIR="/scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0"

echo "================================"
echo "Testing VLM Model Integration"
echo "================================"
echo "Running 1 epoch with small batch size..."
echo ""

# Run with small batch size and 1 epoch for quick validation
python train.py \
    --data_dir $DATA_DIR \
    --batch_size 4 \
    --lr 1e-4 \
    --max_epochs 1

echo ""
echo "================================"
if [ $? -eq 0 ]; then
    echo "✓ Training test passed!"
else
    echo "✗ Training test failed!"
    exit 1
fi
echo "================================"
