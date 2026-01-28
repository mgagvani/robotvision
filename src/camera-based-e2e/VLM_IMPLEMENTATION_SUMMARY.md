# VLM Waypoint Prediction - Implementation Summary

## Overview

Successfully implemented a **frozen VLM-based waypoint prediction model** for the Waymo Open dataset. The model uses SmolVLM-Base (2.1B parameters) as a frozen visual encoder with a small trainable MLP head for waypoint prediction.

---

## What Was Implemented

### 1. Dependencies
✅ Installed `transformers` and `accelerate` libraries

### 2. VLM Model Implementation
✅ **File**: `models/vlm_waypoint_model.py`

**Architecture**:
```
Input Images (1280×1920) 
    ↓
Resize & Normalize (384×384)
    ↓
Frozen SmolVLM Vision Encoder (2.1B params, bfloat16)
    ↓
Visual Features (1152-dim)
    ↓
Concatenate with [Past States (96-dim) + Intent (3-dim)]
    ↓
Trainable MLP Head (926K params)
    ↓
Waypoints (40 values → reshape to 20×2)
```

**Key Features**:
- Only **0.04%** of parameters are trainable (926K out of 2.1B)
- VLM remains frozen in eval mode
- Automatic bfloat16 handling for VLM
- ImageNet normalization for images

### 3. Integration
✅ Updated `train.py` to use VLM model
✅ Updated `models/__init__.py` to export VLM model
✅ Fixed `loader.py` to use local protos

### 4. Testing
✅ Created `test_vlm.py` - verifies VLM loads correctly
✅ Model instantiation and forward pass tested successfully
✅ Training pipeline integration verified

---

## Files Created/Modified

### New Files:
1. **`models/vlm_waypoint_model.py`** - Main VLM model implementation (190 lines)
2. **`test_vlm.py`** - VLM loading verification script
3. **`run_vlm_training.sh`** - SLURM training script
4. **`test_training.sh`** - Quick integration test
5. **`models/__init__.py`** - Module exports

### Modified Files:
1. **`train.py`** - Uses VLM model instead of SAM (lines 21, 42-44)
2. **`loader.py`** - Fixed proto import (line 3)

---

## How to Use

### Quick Test (Interactive)
```bash
# Load environment
module load conda
conda activate waymo_dataloader
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run 1 epoch test
python train.py \
    --data_dir /scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
    --batch_size 4 \
    --lr 1e-4 \
    --max_epochs 1
```

### Full Training (SLURM)
```bash
# Make script executable
chmod +x run_vlm_training.sh

# Submit job
sbatch run_vlm_training.sh

# Monitor job
squeue -u $USER

# Check output
tail -f slurm-<job_id>.out
```

---

## Model Details

### Trainable Parameters
```
Frozen VLM:        2,145,603,440 params (99.96%)
Trainable Head:          926,248 params (0.04%)
────────────────────────────────────────────
Total:             2,146,529,688 params
```

### MLP Head Architecture
```python
Sequential(
    Linear(1251 → 512)     # 1152 (VLM) + 96 (past) + 3 (intent)
    LayerNorm(512)
    GELU()
    Dropout(0.1)
    Linear(512 → 512)
    LayerNorm(512)
    GELU()
    Dropout(0.1)
    Linear(512 → 40)       # 20 waypoints × 2 coords
)
```

### Input/Output Shapes
- **Input**:
  - `PAST`: (B, 16, 6) - past trajectory
  - `IMAGES`: list of 6 cameras, uses index 1 (front camera)
  - `INTENT`: (B,) - values 1, 2, or 3
- **Output**: (B, 40) - flattened waypoints, reshaped to (B, 20, 2)

---

## Training Configuration

### Current Settings (in `run_vlm_training.sh`)
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Epochs**: 10
- **Precision**: bfloat16 mixed
- **Data Subset**: 250K train, 50K val samples

### Expected Resource Usage
- **GPU Memory**: ~15-20GB (VLM: 8-10GB + training: 7-10GB)
- **Training Time**: ~6-8 hours for 10 epochs (depends on GPU)
- **GPU Type**: Works on NVIDIA A30 (tested)

---

## Key Implementation Details

### 1. Image Preprocessing
```python
# Resize from 1280×1920 → 384×384
# Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# Convert to bfloat16 for VLM
```

### 2. VLM Feature Extraction
```python
# Extract from vision_model
outputs = vlm.vision_model(pixel_values=images)
# Pool over spatial patches
features = outputs.last_hidden_state.mean(dim=1)  # (B, 1152)
```

### 3. Protobuf Workaround
Set this environment variable to avoid version conflicts:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 4. No torch.compile
Initially disabled to avoid complications with frozen models. Can be enabled later if needed.

---

## Expected Results

### Week 1 Baseline Goals:
- ✅ Model trains without errors
- ⏳ Validation loss decreases over training
- ⏳ Final validation ADE competitive with SAM baseline (within ±20%)
- ⏳ Trained checkpoint ready for Week 2 (LoRA fine-tuning)

### Comparison with SAM Baseline:
After training completes, compare:
1. **Validation ADE** (Average Displacement Error)
2. **Training time per epoch**
3. **Memory usage**
4. **Loss convergence behavior**

---

## Next Steps (Week 2 - Not Yet Implemented)

After baseline is established:
1. **Add LoRA fine-tuning** to the VLM layers
2. **Experiment with different ranks** (8, 16, 32)
3. **Compare frozen vs LoRA** performance
4. **Try InternVL3_5-1B-Flash** as alternative VLM
5. **Layer ablation studies** if time permits

---

## Troubleshooting

### Issue: "FileNotFoundError" for data
**Solution**: Use correct data path with subdirectory:
```bash
/scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0
```

### Issue: "Descriptors cannot be created directly"
**Solution**: Set protobuf environment variable:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### Issue: Out of Memory
**Solution**: Reduce batch size:
```bash
python train.py --batch_size 4  # or even 2
```

### Issue: "Could not find module Idefics3ImageProcessor"
**Status**: Expected warning, can be ignored. Model works without the processor.

---

## Files Location

```
/home/mathur91/robotvision/src/camera-based-e2e/
├── models/
│   ├── vlm_waypoint_model.py   ← Main implementation
│   ├── __init__.py              ← Updated
│   ├── base_model.py            ← Unchanged
│   └── monocular.py             ← Unchanged (SAM baseline)
├── train.py                     ← Modified to use VLM
├── loader.py                    ← Fixed proto import
├── test_vlm.py                  ← VLM verification
├── run_vlm_training.sh          ← SLURM training script
└── test_training.sh             ← Quick test script
```

---

## Summary

The VLM implementation is **complete and ready for training**. All code has been tested for:
- ✅ Model instantiation
- ✅ Forward pass
- ✅ Integration with PyTorch Lightning
- ✅ Data loading compatibility

**To start training**: Simply run `sbatch run_vlm_training.sh`

The frozen VLM baseline will establish whether pretrained vision-language models provide better features than domain-specific vision models (SAM) for waypoint prediction.
