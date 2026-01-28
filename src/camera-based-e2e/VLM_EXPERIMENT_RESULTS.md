# VLM Waypoint Prediction: Frozen Baseline Experiment

## Overview

This experiment explores "hijacking" a Vision-Language Model (VLM) for waypoint prediction in autonomous driving. Instead of using the VLM's language modeling capabilities, we extract visual features from a frozen VLM and train a small MLP head to predict future waypoints.

## Approach

### Architecture

- **Base Model**: SmolVLM-Base (HuggingFaceTB/SmolVLM-Base, 2.1B parameters)
- **Feature Extraction**: Frozen VLM vision encoder → mean pooled hidden states (1152-dim)
- **Trainable Head**: 3-layer MLP (1251 → 512 → 512 → 40)
- **Input Features**:
  - Visual features from front camera (1152-dim)
  - Past vehicle states (16 timesteps × 6 features = 96-dim)
  - Intent signal (one-hot, 3-dim: left/right/straight)
- **Output**: 20 future waypoints × 2 (X, Y) = 40 values

### Trainable Parameters

| Component | Parameters |
|-----------|------------|
| VLM (frozen) | 2.1B (0 trainable) |
| MLP Head | 926,248 |
| **Total Trainable** | **926,248** |

### Training Configuration

- **Dataset**: Waymo Open End-to-End Driving (250K train, 40K val samples)
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Precision**: bfloat16 mixed precision
- **Epochs**: 10
- **Hardware**: NVIDIA A10 GPU

## Results

### VLM Baseline Performance

| Epoch | Train Loss | Val Loss (ADE) |
|-------|------------|----------------|
| 0 | 1.55 | 1.63 |
| 1 | 1.52 | 1.63 |
| 2 | 1.52 | 1.58 |
| 3 | 1.51 | 1.55 |
| 4 | 1.51 | 1.52 |
| 5 | 1.51 | 1.52 |
| 6 | 1.51 | 1.52 |
| 7 | 1.51 | 1.50 |
| 8 | 1.52 | 1.52 |
| 9 | 1.53 | **1.50** |

**Best Validation ADE: 1.50 meters** (Epoch 9)

### Comparison with SAM Baseline

| Model | Best Val ADE | Trainable Params |
|-------|--------------|------------------|
| SAM Features + MLP | 2.12 m | ~1M |
| **VLM (SmolVLM) + MLP** | **1.50 m** | 926K |

**Improvement: 29% reduction in ADE** (0.62m absolute improvement)

### Learning Curve Analysis

- **Fast convergence**: Most learning happens in epochs 0-4
- **Plateau observed**: Epochs 5-9 show minimal improvement (<0.05m)
- **Recommendation**: Future runs can use 3-5 epochs with early stopping

## Key Findings

1. **VLM features outperform SAM features** for waypoint prediction despite similar MLP head sizes
2. **Frozen VLM works well** - no fine-tuning of the 2.1B parameter backbone needed for strong baseline
3. **Training efficiency**: Model converges quickly due to small trainable capacity and fixed features
4. **Practical ADE of 1.5m** is a reasonable baseline for further improvement with LoRA fine-tuning

## Files

- Model: `models/vlm_waypoint_model.py`
- Training script: `run_vlm_training.sh`
- Best checkpoint: `/scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/checkpoints/camera-e2e-epoch=09-val_loss=1.50.ckpt`
- Training log: `slurm-10229992.out`

## Next Steps

1. **Add LoRA fine-tuning** to unlock VLM adaptation
2. **Multi-camera fusion** - currently only using front camera
3. **Early stopping** to reduce unnecessary training time
4. **Hyperparameter tuning** (learning rate, MLP depth/width)
