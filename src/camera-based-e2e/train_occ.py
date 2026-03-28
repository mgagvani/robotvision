import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from pathlib import Path
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import SimpleProfiler
from torch.utils.data import DataLoader

from models.base_model import LitModel, collate_with_images
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures

from loader import WaymoE2E  # Adjust if using nuscenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--profile", action="store_true", help="Run profiler")
    parser.add_argument("--occ_root", type=str, required=True, help="Path to occupancy root directory")
    args = parser.parse_args()

    # Fix randomness
    pl.seed_everything(42, workers=True)

    # Dataset
    train_dataset = WaymoE2E(indexFile="index_train.pkl", data_dir=args.data_dir, n_items=50000, occ_root=args.occ_root,)
    val_dataset = WaymoE2E(indexFile="index_val.pkl", data_dir=args.data_dir, n_items=5000, occ_root=args.occ_root,)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    # Model
    in_dim = 16 * 6
    out_dim = 20 * 2

    model = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=out_dim,
        n_blocks=4,
    )

    if args.compile:
        model = torch.compile(model, mode="max-autotune")

    lit_model = LitModel(model=model, lr=args.lr)

    # Training
    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    torch.set_float32_matmul_precision("medium")

    # Save checkpoints only
    base_path = Path(args.data_dir).parent
    ckpt_dir = base_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=False,  # No logger
        strategy=strategy,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 16,
        # profiler=SimpleProfiler(extended=True) if args.profile else None,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=ckpt_dir,
                filename="camera-e2e-{epoch:02d}-{val_loss:.2f}",
            ),
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print("Training complete. Best checkpoint saved in:", ckpt_dir)
