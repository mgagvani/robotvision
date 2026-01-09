import argparse 
from datetime import datetime
import sys
from pathlib import Path
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Optional imports for visualization
try:
    from matplotlib import pyplot as plt
    import pandas as pd
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add both current directory (for local models) and parent directory (for top-level models) to path
current_dir = str(Path(__file__).parent)
parent_dir = str(Path(__file__).parent.parent.parent)

# Add current directory first so local 'models' package takes precedence
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Then add parent directory for access to top-level modules (loader, etc.)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)  # Insert at position 1, after current_dir

# Debug: Print path information
print(f"Script location: {Path(__file__).resolve()}")
print(f"Current directory added to path: {current_dir}")
print(f"Parent directory added to path: {parent_dir}")
print(f"Python path (first 3 entries): {sys.path[:3]}")

# Use wrapper loader that fixes the protobuf import issue
try:
    from loader_wrapper import WaymoE2E
    print("✓ Successfully imported loader_wrapper")
except Exception as e:
    print(f"✗ ERROR importing loader_wrapper: {e}")
    import traceback
    traceback.print_exc()
    raise

# Note: We don't import collate_with_images - we use create_filtered_collate_fn instead
# Note: SAMFeatures is imported conditionally when needed for vision models

# Import ablation models (from experiments/ablation/models/)
try:
    from models.ablation_base_model import AblationBaseModel, AblationLitModel
    print("✓ Successfully imported ablation_base_model")
except Exception as e:
    print(f"✗ ERROR importing ablation_base_model: {e}")
    import traceback
    traceback.print_exc()
    raise

# Note: AblationMonocularModel is imported conditionally when needed for vision models
# (it imports models.monocular which we don't need for base models)

def filter_features(past, use_position, use_velocity, use_acceleration):
    """
    Filter past features based on flags.
    past: (B, 16, 6) with columns [pos_x, pos_y, vel_x, vel_y, accel_x, accel_y]
    Returns: (B, 16, num_features) with selected features
    """
    feature_indices = []
    if use_position:
        feature_indices.extend([0, 1])  # pos_x, pos_y
    if use_velocity:
        feature_indices.extend([2, 3])  # vel_x, vel_y
    if use_acceleration:
        feature_indices.extend([4, 5])  # accel_x, accel_y
    
    if len(feature_indices) == 0:
        raise ValueError("At least one feature type must be enabled")
    
    return past[:, :, feature_indices]

def create_filtered_collate_fn(use_position, use_velocity, use_acceleration, has_images=True):
    """Create a collate function that filters features"""
    def collate_fn(batch):
        past = [torch.as_tensor(b["PAST"], dtype=torch.float32) for b in batch]
        future = [torch.as_tensor(b["FUTURE"], dtype=torch.float32) for b in batch]
        intent = torch.as_tensor([b["INTENT"] for b in batch])
        names = [b["NAME"] for b in batch]

        # Handle images - they may not be present for base models
        if has_images and "IMAGES" in batch[0] and batch[0]["IMAGES"] is not None:
            cams = list(zip(*[b["IMAGES"] for b in batch]))  # per-camera tuples
            images = [torch.stack(cam_imgs, dim=0) for cam_imgs in cams]  # stay on CPU
        else:
            # Create tiny dummy images for base models (they won't be used)
            # Use minimal size to save memory - 8x8 instead of 1280x1920
            batch_size = len(batch)
            images = [torch.zeros((batch_size, 3, 8, 8)) for _ in range(6)]  # Tiny placeholder

        # Stack past and filter features
        past_stacked = torch.stack(past, dim=0)  # (B, 16, 6)
        past_filtered = filter_features(past_stacked, use_position, use_velocity, use_acceleration)

        return {
            "PAST": past_filtered,
            "FUTURE": torch.stack(future, dim=0),
            "INTENT": intent,
            "IMAGES": images,
            "NAME": names,
        }
    return collate_fn

def get_experiment_name(model_type, use_position, use_velocity, use_acceleration, use_intent):
    """Generate experiment name based on features"""
    name_parts = [model_type]
    if use_position:
        name_parts.append("pos")
    if use_velocity:
        name_parts.append("vel")
    if use_acceleration:
        name_parts.append("acc")
    if use_intent:
        name_parts.append("intent")
    return "_".join(name_parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--use_position', action='store_true', help='Include position features')
    parser.add_argument('--use_velocity', action='store_true', help='Include velocity features')
    parser.add_argument('--use_acceleration', action='store_true', help='Include acceleration features')
    parser.add_argument('--use_intent', action='store_true', help='Include intent feature')
    parser.add_argument('--model_type', type=str, choices=['base', 'vision'], required=True, help='Model type: base or vision')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for logs and checkpoints')
    args = parser.parse_args()

    # Validate at least one feature is selected
    if not (args.use_position or args.use_velocity or args.use_acceleration):
        raise ValueError("At least one of --use_position, --use_velocity, or --use_acceleration must be set")

    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)  # For data sampling in loader

    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = get_experiment_name(
            args.model_type, 
            args.use_position, 
            args.use_velocity, 
            args.use_acceleration, 
            args.use_intent
        )

    # Data 
    has_images = (args.model_type == 'vision')
    print(f"\nLoading datasets (images={'enabled' if has_images else 'disabled'})...")
    train_dataset = WaymoE2E(batch_size=args.batch_size, indexFile='index_train.pkl', data_dir=args.data_dir, images=has_images, n_items=25000)
    test_dataset = WaymoE2E(batch_size=args.batch_size, indexFile='index_val.pkl', data_dir=args.data_dir, images=has_images, n_items=5000)
    print(f"✓ Datasets loaded: {len(train_dataset)} training samples, {len(test_dataset)} validation samples")

    # Create filtered collate function
    collate_fn = create_filtered_collate_fn(args.use_position, args.use_velocity, args.use_acceleration, has_images=has_images)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=2, 
        collate_fn=collate_fn, 
        persistent_workers=False, 
        pin_memory=False,
        # Note: shuffle is not allowed with IterableDataset - shuffling is handled by the dataset itself
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=2, 
        collate_fn=collate_fn, 
        persistent_workers=False, 
        pin_memory=False,
        # Note: shuffle is not allowed with IterableDataset
    )

    # Calculate input dimension
    num_features = sum([args.use_position * 2, args.use_velocity * 2, args.use_acceleration * 2])
    in_dim = 16 * num_features  # Past: (B, 16, num_features)
    out_dim = 20 * 2  # Future: (B, 20, 2)

    print(f"\nModel configuration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Input dimension: {in_dim}")
    print(f"  Output dimension: {out_dim}")
    print(f"  Number of features: {num_features}")
    print(f"  Features: position={args.use_position}, velocity={args.use_velocity}, acceleration={args.use_acceleration}, intent={args.use_intent}")

    # Model
    print(f"\nInitializing model...")
    if args.model_type == 'base':
        model = AblationBaseModel(in_dim=in_dim, out_dim=out_dim)
        lit_model = AblationLitModel(model=model, lr=args.lr)
        print(f"✓ Base model initialized")
    else:  # vision
        # Import vision model components only when needed
        try:
            # First, import SAMFeatures from the TOP-LEVEL models package
            # (not from the local experiments/ablation/models package)
            # We need to manipulate sys.path temporarily to ensure we get the right package
            original_syspath = sys.path.copy()
            local_models_pkg = sys.modules.get('models', None)
            
            # Clear models-related modules to force fresh import
            models_to_clear = [k for k in list(sys.modules.keys()) if k.startswith('models.') or k == 'models']
            cleared_modules = {k: sys.modules.pop(k) for k in models_to_clear if k in sys.modules}
            
            # Temporarily prioritize parent_dir (remove ablation dir from path)
            sys.path = [p for p in sys.path if 'experiments/ablation' not in p]
            
            # Import from top-level models package
            import models as toplevel_models
            from models.monocular import SAMFeatures
            print("✓ Successfully imported SAMFeatures from top-level models")
            
            # Restore sys.path
            sys.path = original_syspath
            
            # Restore local models package for importing ablation models
            if local_models_pkg is not None:
                sys.modules['models'] = local_models_pkg
            else:
                # Clear top-level models from 'models' key
                if 'models' in sys.modules:
                    del sys.modules['models']
            
            # Now import the ablation model (from local models package)
            from models.ablation_monocular import AblationMonocularModel
            print("✓ Successfully imported AblationMonocularModel")
            
        except Exception as e:
            # Restore sys.path on error
            if 'original_syspath' in locals():
                sys.path = original_syspath
            print(f"✗ ERROR importing vision model components: {e}")
            print(f"Current Python path: {sys.path[:5]}")
            print(f"Looking for models at: {Path(parent_dir) / 'models'}")
            print(f"Models directory exists: {(Path(parent_dir) / 'models').exists()}")
            import traceback
            traceback.print_exc()
            raise
        
        model = AblationMonocularModel(
            in_dim=in_dim, 
            out_dim=out_dim, 
            feature_extractor=SAMFeatures(),
            use_intent=args.use_intent
        )
        lit_model = AblationLitModel(model=model, lr=args.lr)
        print(f"✓ Vision model initialized")

    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"
    visualizations_dir = output_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    print(f"\nSetting up trainer...")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Logs directory: {logs_dir}")
    print(f"  Checkpoints directory: {checkpoints_dir}")
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        # accelerator='gpu' is default when GPU is available, 'cpu' for debugging
        logger=CSVLogger(str(logs_dir), name=args.experiment_name),
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                mode='min', 
                save_top_k=1, 
                dirpath=str(checkpoints_dir),
                filename=f'{args.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}'
            ),
        ],
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=50,
    )

    print(f"\n{'='*60}")
    print(f"Starting training for: {args.experiment_name}")
    print(f"{'='*60}\n")
    
    try:
        trainer.fit(lit_model, train_loader, val_loader)
        print(f"\n{'='*60}")
        print(f"✓ Training completed successfully for: {args.experiment_name}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERROR: Training failed for {args.experiment_name}")
        print(f"Error: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to ensure exit code is non-zero

    # Export loss graph
    if HAS_VISUALIZATION:
        print(f"\nGenerating loss plot...")
        try:
            exp_dir = logs_dir / args.experiment_name
            if not exp_dir.exists():
                print(f"Warning: Experiment directory {exp_dir} not found, skipping plot generation")
            else:
                # Find the latest version directory
                version_dirs = sorted(exp_dir.glob("version_*"), key=lambda x: int(x.name.split("_")[1]))
                if not version_dirs:
                    print(f"Warning: No version directories found in {exp_dir}, skipping plot generation")
                else:
                    run_dir = version_dirs[-1]  # Use latest version
                    metrics = pd.read_csv(run_dir / "metrics.csv")
                    train = metrics[metrics["train_loss"].notna()]
                    val = metrics[metrics["val_loss"].notna()]

                    if len(train) > 0 and len(val) > 0:
                        plt.figure()
                        plt.plot(train["step"], train["train_loss"], label="train_loss")
                        plt.plot(val["step"], val["val_loss"], label="val_loss")
                        plt.xlabel("Step")
                        plt.ylabel("Loss")
                        plt.title(f"Loss: {args.experiment_name}")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(visualizations_dir / f"{args.experiment_name}_loss.png", dpi=200)
                        print(f"✓ Loss plot saved to {visualizations_dir / f'{args.experiment_name}_loss.png'}")
                        
                        # Print final metrics
                        final_train_loss = train["train_loss"].iloc[-1]
                        final_val_loss = val["val_loss"].iloc[-1]
                        min_val_loss = val["val_loss"].min()
                        print(f"\nFinal metrics:")
                        print(f"  Final training loss: {final_train_loss:.4f}")
                        print(f"  Final validation loss: {final_val_loss:.4f}")
                        print(f"  Minimum validation loss: {min_val_loss:.4f}")
                    else:
                        print(f"Warning: No training/validation data found for plotting")
        except Exception as e:
            print(f"Error generating plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipping plot generation (matplotlib/pandas not available)")

