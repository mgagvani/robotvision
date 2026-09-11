def _monkeypatch_pillow_version() -> None:
    """Torchmetrics expects PIL.__version__; some Pillow builds only expose PILLOW_VERSION."""
    try:
        import PIL
    except ImportError:
        return
    if hasattr(PIL, "__version__"):
        return
    version = getattr(PIL, "PILLOW_VERSION", None)
    if version is None:
        try:
            from importlib.metadata import version as pkg_version

            version = pkg_version("pillow")
        except Exception:
            version = "0.0.0"
    PIL.__version__ = version


_monkeypatch_pillow_version()

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
import os

from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images
from models.drivor import DrivoRModel
from models.feature_extractors import SAMFeatures
from models.gtrs import GTRSModel
from models.monocular import DeepMonocularModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model with torch.compile",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Whether to run the profiler"
    )
    parser.add_argument(
        "--train_items",
        type=int,
        default=250_000,
        help="Number of training samples to use.",
    )
    parser.add_argument(
        "--val_items",
        type=int,
        default=25_000,
        help="Number of validation samples to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepmonocular",
        choices=["deepmonocular", "gtrs", "drivor"],
        help="Model architecture to train",
    )
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    # Data
    train_dataset = WaymoE2E(
        indexFile="index_train.pkl", data_dir=args.data_dir, n_items=args.train_items
    )
    test_dataset = WaymoE2E(
        indexFile="index_val.pkl", data_dir=args.data_dir, n_items=args.val_items
    )
    nw = 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=nw,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=nw,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    # Model
    out_dim = 20 * 2  # Future: (B, 20, 2)
    feature_extractor = SAMFeatures(
        model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
    )
    if args.model == "gtrs":
        model = GTRSModel(feature_extractor=feature_extractor, out_dim=out_dim)
    elif args.model == "drivor":
        model = DrivoRModel(feature_extractor=feature_extractor, out_dim=out_dim)
    else:
        model = DeepMonocularModel(feature_extractor=feature_extractor, out_dim=out_dim)
    name = args.model
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    lit_model = LitModel(model=model, lr=args.lr)

    # We don't want to save logs or checkpoints in the home directory - it'll fill up fast
    base_path = Path(args.data_dir).parent.as_posix()
    sample_tag = f"n{args.train_items}"
    timestamp = f"{name}_e2e_waymo_{sample_tag}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb_logger = WandbLogger(
        name=timestamp,
        save_dir=base_path + "/logs",
        project="robotvision",
        log_model=True,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        wandb_logger.watch(lit_model, log="all")

    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=[CSVLogger(base_path + "/logs", name=timestamp), wandb_logger],
        strategy=strategy,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 16,
        log_every_n_steps=10,
        profiler=SimpleProfiler(extended=True) if args.profile else None,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=base_path + "/checkpoints",
                filename=f"camera-e2e-{name}-waymo-{sample_tag}-{{epoch:02d}}-{{val_loss:.2f}}",
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
