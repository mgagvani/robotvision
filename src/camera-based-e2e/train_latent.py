import argparse
from datetime import datetime
from pytorch_lightning.callbacks import TQDMProgressBar

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from matplotlib import pyplot as plt
import pandas as pd

import torch
from pathlib import Path

from loader import WaymoE2E

# Replace with your model defined in models/
from models.transfuser.team_code_transfuser.latentTF import LatentTFModel
from models.base_model import collate_with_images
from models.transfuser.team_code_transfuser.config import GlobalConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Waymo E2E data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--train_index", type=str, default="index_train.pkl",
                    help="Train index .pkl (default: index_train.pkl)")
    args = parser.parse_args()

    # Config
    config = GlobalConfig(setting='eval')

    # Data
    train_dataset = WaymoE2E(
        batch_size=args.batch_size,
        indexFile=args.train_index,
        data_dir=args.data_dir,
        images=True,
        n_items=25000,
    )
    val_dataset = WaymoE2E(
        batch_size=args.batch_size,
        indexFile="index_val.pkl",
        data_dir=args.data_dir,
        images=True,
        n_items=5000,
    )

    test_dataset = WaymoE2E(
        batch_size=args.batch_size,
        indexFile="index_test.pkl",
        data_dir=args.data_dir,
        images=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    # all_intents = []
    # for batch in train_loader:
    #     all_intents.append(batch["INTENT"])
    # all_intents = torch.cat(all_intents)
    # print("ALL UNIQUE INTENT VALUES:", all_intents.unique())

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    # Model
    lit_model = LatentTFModel(
        config=config,
        lr=args.lr,
        image_architecture='resnet34',
        lidar_architecture='resnet18',
        use_velocity=True
    )

    base_path = Path(args.data_dir).parent.as_posix()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger(base_path + "/logs", name=f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"),
        precision="bf16-mixed",
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=base_path + "/checkpoints",
                filename="latent-tf-{epoch:02d}-{val_loss:.2f}",
            ),
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # # Export loss graph to visualizations/
    # try:
    #     base_path = Path(base_path)
    #     run_dir = sorted((base_path / "logs").glob("camera_e2e_*"))[-1]  # newest run
    #     metrics = pd.read_csv(run_dir / "version_0" / "metrics.csv")
    #     train = metrics[metrics["train_loss"].notna()]
    #     val = metrics[metrics["val_loss"].notna()]

    #     plt.figure()
    #     plt.plot(train["step"], train["train_loss"], label="train_loss")
    #     plt.plot(val["step"], val["val_loss"], label="val_loss")
    #     plt.xlabel("Step")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.tight_layout()
    #     out = Path("./visualizations")
    #     plt.savefig(out / "loss.png", dpi=200)
    # except Exception as e:
    #     print(f"Could not save loss plot: {e}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    trainer.test(lit_model, dataloaders=test_loader)