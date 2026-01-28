# VLM Waypoint Prediction - Quick Start Guide

## Implementation Status: ✅ COMPLETE

All code is implemented and ready for training!

---

## What's Been Done (Day 1)

✅ **Installed dependencies**: `transformers`, `accelerate`  
✅ **Created VLM model**: `models/vlm_waypoint_model.py` (frozen SmolVLM + trainable head)  
✅ **Integrated with training**: Updated `train.py` to use VLM  
✅ **Fixed compatibility issues**: Protobuf, data loader imports  
✅ **Tested model**: Verified loading and forward pass work  
✅ **Created training scripts**: Ready-to-use SLURM script  

**Model Details**:
- Frozen VLM: 2.1B parameters (SmolVLM-Base)
- Trainable Head: 926K parameters (MLP)
- Total trainable: **0.04%** of model

---

## How to Start Training (2 options)

### Option 1: SLURM Job (Recommended)
```bash
cd /home/mathur91/robotvision/src/camera-based-e2e
sbatch run_vlm_training.sh
```

Monitor progress:
```bash
squeue -u $USER                    # Check job status
tail -f slurm-<job_id>.out        # Watch training logs
```

### Option 2: Interactive (for testing)
```bash
module load conda
conda activate waymo_dataloader
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /home/mathur91/robotvision/src/camera-based-e2e

# Quick test (1 epoch, small batch)
python train.py \
    --data_dir /scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
    --batch_size 4 \
    --lr 1e-4 \
    --max_epochs 1
```

---

## Training Configuration

Current settings in `run_vlm_training.sh`:
- **Batch size**: 16 (adjust if OOM)
- **Learning rate**: 1e-4
- **Epochs**: 10
- **Data**: 250K train samples, 50K val samples

Expected time: **~6-8 hours** on A30 GPU

---

## After Training Completes

### 1. Check Results
```bash
# View loss plot
eog visualizations/loss.png

# Check final metrics
tail slurm-<job_id>.out
```

### 2. Compare with SAM Baseline
Look for validation ADE (Average Displacement Error):
- **VLM model**: Check your training output
- **SAM baseline**: Check previous training logs

### 3. Analyze Checkpoint
```bash
ls -lh /scratch/gilbreth/mathur91/waymo_end_to_end_camera_v1_0_0/checkpoints/
```

---

## File Structure

```
camera-based-e2e/
├── models/
│   └── vlm_waypoint_model.py      ← New VLM implementation
├── train.py                        ← Modified (uses VLM)
├── loader.py                       ← Fixed (proto import)
├── run_vlm_training.sh            ← SLURM script (run this!)
├── VLM_IMPLEMENTATION_SUMMARY.md  ← Detailed docs
└── QUICK_START.md                 ← This file
```

---

## Troubleshooting

**Q: Out of memory?**  
A: Reduce batch size to 8 or 4 in `run_vlm_training.sh`

**Q: Training seems stuck?**  
A: First epoch takes longer (model loading). Check `nvidia-smi` to verify GPU usage.

**Q: How to compare with SAM?**  
A: After VLM training finishes, check the final validation loss and compare with previous SAM training runs.

---

## Next Steps (Week 2 - After Baseline)

Once you have baseline results:
1. Add LoRA fine-tuning (unfreeze some VLM layers)
2. Try different LoRA ranks (8, 16, 32)
3. Experiment with InternVL3_5-1B-Flash
4. Optimize hyperparameters if needed

---

## Summary

🎯 **Everything is ready!** Just run:
```bash
sbatch run_vlm_training.sh
```

The training will:
1. Load frozen SmolVLM (2.1B params)
2. Train only the MLP head (926K params)
3. Save checkpoints and logs
4. Generate loss plots

Check back in ~6-8 hours for results! 🚀
