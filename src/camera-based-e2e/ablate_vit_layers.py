import argparse
import gc
import random
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import LightningEnvironment

from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images
from models.monocular import MonocularModel, ViTIntermediateFeatures


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")


def make_loader(dataset: WaymoE2E, batch_size: int, num_workers: int, seed: int):
    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def smooth_train_loss(metrics_path: Path, window: int) -> Optional[float]:
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    train = metrics[metrics["train_loss"].notna()]
    if train.empty:
        return None
    return float(train["train_loss"].tail(window).mean())


def extract_val_losses(metrics_path: Path) -> tuple[Optional[float], Optional[float]]:
    if not metrics_path.exists():
        return None, None
    metrics = pd.read_csv(metrics_path)
    val = metrics[metrics["val_loss"].notna()]
    if val.empty:
        return None, None
    return float(val["val_loss"].iloc[-1]), float(val["val_loss"].min())


def plot_results(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(df["layer"], df["train_loss"], label="train_loss (tail mean)")
    plt.plot(df["layer"], df["val_last"], label="val_loss (last)")
    plt.plot(df["layer"], df["val_best"], label="val_loss (best)", linestyle="--")
    plt.xlabel("Transformer block index")
    plt.ylabel("ADE loss")
    plt.title(f"ViT layer ablation: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_single_layer(
    layer_idx: int,
    args,
    log_root: Path,
) -> dict:
    set_seed(args.seed)

    feature_extractor = ViTIntermediateFeatures(
        model_name=args.model_name,
        layer_indices=[layer_idx],
        frozen=not args.unfreeze_features,
        use_cls_token=args.use_cls_token,
    )
    model = MonocularModel(
        in_dim=16 * 6,
        out_dim=20 * 2,
        feature_extractor=feature_extractor,
    )
    if args.compile:
        model = torch.compile(model, mode="max-autotune")

    lit_model = LitModel(model=model, lr=args.lr)

    train_dataset = WaymoE2E(
        batch_size=args.batch_size,
        indexFile="index_train.pkl",
        data_dir=args.data_dir,
        images=True,
        n_items=args.n_train_items,
        seed=args.seed,
    )
    val_dataset = WaymoE2E(
        batch_size=args.batch_size,
        indexFile="index_val.pkl",
        data_dir=args.data_dir,
        images=True,
        n_items=args.n_val_items,
        seed=args.seed,
    )

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, args.seed)

    log_name = f"vit_{sanitize_name(args.model_name)}"
    logger = CSVLogger(
        save_dir=log_root,
        name=log_name,
        version=f"layer_{layer_idx}",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = args.precision if accelerator == "gpu" else 32

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else 1,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        precision=precision,
        enable_checkpointing=False,
        deterministic=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        plugins=[LightningEnvironment()],
    )

    trainer.fit(lit_model, train_loader, val_loader)

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    train_loss = smooth_train_loss(metrics_path, args.train_loss_window)
    val_last, val_best = extract_val_losses(metrics_path)

    # cleanup between runs to avoid memory accumulation
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "layer": layer_idx,
        "train_loss": train_loss,
        "val_last": val_last,
        "val_best": val_best,
        "log_dir": str(logger.log_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablate ViT intermediate layers for E2E planning.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Waymo E2E data directory.")
    parser.add_argument("--model_name", type=str, default="vit_small_plus_patch16_dinov3.lvd1689m", help="TIMM ViT model id.")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Specific layer indices to evaluate (0-based).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=int, default=300, help="Number of train batches per layer (int for determinism).")
    parser.add_argument("--limit_val_batches", type=int, default=60, help="Number of val batches per layer.")
    parser.add_argument("--n_train_items", type=int, default=2048, help="Subset size for training examples.")
    parser.add_argument("--n_val_items", type=int, default=512, help="Subset size for validation examples.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1, help="Number of devices for Lightning trainer.")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the planning model.")
    parser.add_argument("--use_cls_token", action="store_true", help="Keep cls/dist tokens instead of dropping them.")
    parser.add_argument("--unfreeze_features", action="store_true", help="Train the ViT backbone weights.")
    parser.add_argument("--train_loss_window", type=int, default=20, help="Tail window (steps) used to smooth train loss.")
    parser.add_argument("--log_dir", type=str, default="ablation_logs")
    parser.add_argument("--out_dir", type=str, default="visualizations")

    args = parser.parse_args()

    set_seed(args.seed)

    # decide which layers to sweep
    if args.layers is None:
        probe = ViTIntermediateFeatures(
            model_name=args.model_name,
            layer_indices=None,
            frozen=True,
            use_cls_token=args.use_cls_token,
        )
        layers = probe.layer_indices
        del probe
    else:
        layers = args.layers

    log_root = Path(args.log_dir)
    out_root = Path(args.out_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    for layer_idx in layers:
        metrics = run_single_layer(layer_idx, args, log_root)
        results.append(metrics)

    df = pd.DataFrame(results).sort_values("layer")
    csv_path = out_root / f"vit_layer_ablation_{sanitize_name(args.model_name)}.csv"
    df.to_csv(csv_path, index=False)

    plot_path = out_root / f"vit_layer_ablation_{sanitize_name(args.model_name)}.png"
    plot_results(df, args.model_name, plot_path)

    print(f"Ablation completed. Results saved to {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
