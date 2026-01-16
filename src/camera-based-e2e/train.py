import argparse 
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle

from preference_loader import WaymoE2E
from torch.utils.data import random_split, DataLoader

# Replace with your model defined in models/ 
from models.base_model import LitModel, collate_with_images
from models.monocular import MonocularModel, DeepMonocularModel, SAMFeatures
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # Data 
    # Initialize the full dataset
    full_dataset = WaymoE2E(
        indexFile='preference_cache.pkl',
        data_dir=args.data_dir,
        images=True,
        n_items=250000
    )

    # Decide split sizes
    train_size = int(0.8 * len(full_dataset))  # 80% train
    val_size = len(full_dataset) - train_size  # 20% val

    # Random split
    # Load full dataset indexes
    with open("preference_cache.pkl", "rb") as f:
        indexes = pickle.load(f)

    # Shuffle for randomness
    import random
    random.seed(42)
    random.shuffle(indexes)

    # Split 80/20
    split = int(0.8 * len(indexes))
    train_indexes = indexes[:split]
    val_indexes = indexes[split:]

    # Create train / val datasets
    train_dataset = WaymoE2E(indexFile='preference_cache.pkl', data_dir=args.data_dir, images=True)
    train_dataset.indexes = train_indexes  # override indexes

    val_dataset = WaymoE2E(indexFile='preference_cache.pkl', data_dir=args.data_dir, images=True)
    val_dataset.indexes = val_indexes  # override indexes
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=12, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=12, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)

    # Model
    in_dim = 16 * 6 + 3 + 21 * 2 
    out_dim = 1  

    model = DeepMonocularModel(out_dim=out_dim, feature_extractor=SAMFeatures(model_name="timm/vit_pe_spatial_tiny_patch16_512.fb"))
    lit_model = LitModel(model=torch.compile(model, mode="max-autotune"), lr=args.lr)

    base_path = Path(args.data_dir).parent.as_posix()
    # We don't want to save logs or checkpoints in the home directory - it'll fill up fast
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger(base_path + "/logs", name=f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"),
        precision="bf16-mixed",
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





    