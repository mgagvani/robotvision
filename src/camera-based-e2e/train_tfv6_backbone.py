import argparse
from datetime import datetime
import pickle
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import SimpleProfiler

from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images
from models.tfv6_backbone_model import TFv6BackboneTrajectoryModel

def resolve_index_path(index_arg: str, data_dir: str) -> str:
    candidate_paths = [
        Path(index_arg),
        Path(__file__).resolve().parent / index_arg,
        Path(data_dir) / index_arg,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.as_posix()
    return index_arg


def validate_index_files(index_path: str, data_dir: str, sample_limit: int = 128) -> None:
    with open(index_path, "rb") as f:
        index = pickle.load(f)

    checked = set()
    missing = []
    for filename, _, _ in index:
        if filename in checked:
            continue
        checked.add(filename)
        if not (Path(data_dir) / filename).exists():
            missing.append(filename)
        if len(checked) >= sample_limit:
            break

    if missing:
        missing_examples = "\n".join(f"  - {name}" for name in missing[:5])
        raise RuntimeError(
            "Index/data_dir mismatch detected.\n"
            f"index: {index_path}\n"
            f"data_dir: {data_dir}\n"
            f"Checked {len(checked)} unique shard names and found {len(missing)} missing.\n"
            "Missing examples:\n"
            f"{missing_examples}\n"
            "Use a data_dir containing raw Waymo tfrecord shards that match this index."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Waymo E2E data directory")
    parser.add_argument("--train_index", type=str, default="index_train.pkl", help="Path to train index pickle")
    parser.add_argument("--val_index", type=str, default="index_val.pkl", help="Path to val index pickle")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--n_train_items", type=int, default=250_000, help="Training subset size")
    parser.add_argument("--n_val_items", type=int, default=25_000, help="Validation subset size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor when num_workers > 0")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Lightning logging frequency")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validation frequency in epochs")
    parser.add_argument("--disable_progress_bar", action="store_true", help="Disable progress bar to reduce stdout overhead")
    parser.add_argument("--rfs_weight", type=float, default=0.0, help="RFS loss weight in LitModel")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze TFv6 backbone weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root for logs/checkpoints")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model with torch.compile")
    parser.add_argument("--profile", action="store_true", help="Whether to run the profiler")
    args = parser.parse_args()

    train_index = resolve_index_path(args.train_index, args.data_dir)
    val_index = resolve_index_path(args.val_index, args.data_dir)
    validate_index_files(train_index, args.data_dir)
    validate_index_files(val_index, args.data_dir)

    if torch.cuda.is_available():
        # Use Tensor Cores on A10 for faster matmul kernels.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
    train_dataset = WaymoE2E(
        indexFile=train_index,
        data_dir=args.data_dir,
        n_items=args.n_train_items,
    )
    val_dataset = WaymoE2E(
        indexFile=val_index,
        data_dir=args.data_dir,
        n_items=args.n_val_items,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
    )

    out_dim = 20 * 2  # Future: (B, 20, 2)
    model = TFv6BackboneTrajectoryModel(
        out_dim=out_dim,
        freeze_backbone=args.freeze_backbone,
    )
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    lit_model = LitModel(model=model, lr=args.lr, rfs_weight=args.rfs_weight)

    output_root = Path(args.output_dir) if args.output_dir else Path(args.data_dir).parent
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = f"tfv6_backbone_{datetime.now().strftime('%Y%m%d_%H%M')}"

    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger((output_root / "logs").as_posix(), name=timestamp),
        strategy=strategy,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 16,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        enable_progress_bar=not args.disable_progress_bar,
        profiler=SimpleProfiler(extended=True) if args.profile else None,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=(output_root / "checkpoints").as_posix(),
                filename="tfv6-backbone-{epoch:02d}-{val_loss:.4f}",
            ),
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.resume_ckpt)

    # Explicit end-of-run validation summary for sbatch logs.
    final_val = trainer.validate(lit_model, dataloaders=val_loader, ckpt_path="best")
    if final_val:
        print("Final validation metrics (best checkpoint):")
        for key, value in sorted(final_val[0].items()):
            print(f"  {key}: {value:.6f}")

    # Export loss graph from CSV logs.
    try:
        run_dir = sorted((output_root / "logs").glob("tfv6_backbone_*"))[-1]
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
        viz_dir = output_root / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / "tfv6_backbone_loss.png", dpi=200)
    except Exception as error:
        print(f"Could not save loss plot: {error}")