# Feature Ablation Experiments

This directory contains scripts for running feature ablation experiments to test different combinations of input features (position, velocity, acceleration, intent) across both base and vision models.

## Directory Structure

```
experiments/ablation/
├── train_ablation.py              # Training script with feature selection
├── generate_experiments.py        # Script to generate experiment configs
├── run_ablation_experiments.sbatch # SLURM array job script
├── experiment_configs.json        # Generated experiment configurations
├── models/                        # Ablation-specific model files
│   ├── ablation_base_model.py
│   └── ablation_monocular.py
└── outputs/                       # All experiment outputs
    ├── logs/                      # Training logs per experiment
    ├── checkpoints/               # Model checkpoints
    └── visualizations/            # Loss plots
```

## Experiment Setup

The experiments test 15 feature combinations (excluding empty set) × 2 models = 30 total experiments:

- **Feature types**: position (pos_x, pos_y), velocity (vel_x, vel_y), acceleration (accel_x, accel_y), intent
- **Models**: base (BaseModel) and vision (MonocularModel)

## Results Summary

**Best Performing Models** (by minimum validation loss):
1. `vision_pos_acc` - 2.137 (position + acceleration + camera)
2. `vision_pos_vel_acc` - 2.142 (position + velocity + acceleration + camera)
3. `vision_vel_acc` - 2.146 (velocity + acceleration + camera)

**Key Findings:**
- **Vision models significantly outperform base models** (mean: 2.76 vs 4.02 validation loss)
- **Camera features provide substantial value** when combined with kinematic features
- **Feature hierarchy**: Velocity and/or position are critical; acceleration helps when combined with others; acceleration alone performs poorly (12+ loss)
- **Intent feature issue**: A bug was discovered where the base model did not use the intent feature (extracted but not passed to the model). Vision models do use intent correctly, but show minimal/negative impact when added (~1-5% change), suggesting intent is not predictive for this task in the current encoding
- **Performance saturation**: Among well-performing models (excluding acceleration-only), validation losses cluster tightly (2.13-2.47), suggesting the model architecture may be the limiting factor rather than feature selection

**Recommendation**: Use `vision_pos_acc` or `vision_vel_acc` for best performance with minimal feature complexity. Intent feature can be omitted.

## Usage

### 1. Generate Experiment Configurations

```bash
cd /home/mathur91/robotvision/src/camera-based-e2e/experiments/ablation
python3 generate_experiments.py
```

This creates `experiment_configs.json` with all 30 experiment configurations.

### 2. Run All Experiments (SLURM Array Job)

```bash
sbatch run_ablation_experiments.sbatch
```

This will submit 30 array jobs (one per experiment) that run in parallel.

### 3. Run Single Experiment Manually

```bash
python train_ablation.py \
    --data_dir /path/to/data \
    --batch_size 16 \
    --lr 0.001 \
    --max_epochs 10 \
    --model_type base \
    --use_position \
    --use_velocity \
    --output_dir ./outputs
```

## Command Line Arguments

### Required
- `--data_dir`: Path to Waymo E2E data directory
- `--model_type`: Either "base" or "vision"

### Feature Selection (at least one required)
- `--use_position`: Include position features (pos_x, pos_y)
- `--use_velocity`: Include velocity features (vel_x, vel_y)
- `--use_acceleration`: Include acceleration features (accel_x, accel_y)
- `--use_intent`: Include intent feature

### Optional
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--max_epochs`: Number of epochs (default: 15)
- `--experiment_name`: Custom experiment name (auto-generated if not provided)
- `--output_dir`: Output directory (default: ./outputs)

## Outputs

Each experiment creates:
- **Logs**: CSV logs in `outputs/logs/{experiment_name}_*/`
- **Checkpoints**: Best model in `outputs/checkpoints/{experiment_name}-*.ckpt`
- **Visualizations**: Loss plot in `outputs/visualizations/{experiment_name}_loss.png`

## Experiment Naming

Experiments are automatically named based on included features:
- `base_pos_vel`: Base model with position and velocity
- `vision_pos_vel_acc_intent`: Vision model with all features
- etc.

## Notes

- All outputs are stored in the `outputs/` subdirectory to keep experiments separate from working code
- The original model files are not modified - all changes are in the ablation-specific model files
- Index files (`index_train.pkl`, `index_val.pkl`) are automatically copied if needed
- **Known Issue**: Base model experiments with intent have identical results to their non-intent counterparts (e.g., `base_pos` = `base_pos_intent`) due to a bug where intent was extracted but not used in the forward pass. This was fixed in the codebase but results remain from the original run. Vision model experiments use intent correctly.


