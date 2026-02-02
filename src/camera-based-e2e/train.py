import argparse 
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import SimpleProfiler

from matplotlib import pyplot as plt
import pandas as pd

import torch
from pathlib import Path

from loader import WaymoE2E

# Replace with your model defined in models/ 
from models.base_model import LitModel, collate_with_images
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model with torch.compile')
    args = parser.parse_args()

    # Data 
    train_dataset = WaymoE2E(indexFile='index_train.pkl', data_dir=args.data_dir, images=True, n_items=100_000)
    test_dataset = WaymoE2E(indexFile='index_val.pkl', data_dir=args.data_dir, images=True, n_items=25_000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=14, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=14, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)

    # Model
    in_dim = 16 * 6  # Past: (B, 16, 6)
    out_dim = 20 * 2  # Future: (B, 20, 2)

    model = DeepMonocularModel(feature_extractor=SAMFeatures(model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True), out_dim=out_dim, n_blocks=4)
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    lit_model = LitModel(model=model, lr=args.lr)

    # We don't want to save logs or checkpoints in the home directory - it'll fill up fast
    base_path = Path(args.data_dir).parent.as_posix()
    timestamp = f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb_logger = WandbLogger(name=timestamp, save_dir=base_path + "/logs", project="robotvision", log_model=True)
    wandb_logger.watch(lit_model, log="all")

    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=[CSVLogger(base_path + "/logs", name=timestamp), wandb_logger],
        strategy=strategy,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 16,
        log_every_n_steps=10,
        profiler=SimpleProfiler(extended=True),
        callbacks=[
            ModelCheckpoint(monitor='val_loss',
                             mode='min', 
                             save_top_k=1, 
                             dirpath=base_path + '/checkpoints',
                             filename='camera-e2e-{epoch:02d}-{val_loss:.2f}'
                            ),
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Export loss graph to visualizations/
    try:
        base_path = Path(base_path)
        run_dir = sorted((base_path / "logs").glob("camera_e2e_*"))[-1]  # newest run
        metrics = pd.read_csv(run_dir / "version_0" / "metrics.csv")
        train = metrics[metrics["train_loss"].notna()]
        val = metrics[metrics["val_loss"].notna()]

        plt.figure()
        plt.plot(train["step"], train["train_loss"], label="train_loss")
        plt.plot(val["step"], val["val_loss"], label="val_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        out = Path("./visualizations")
        plt.savefig(out / "loss.png", dpi=200)
    except Exception as e:
        print(f"Could not save loss plot: {e}")





    