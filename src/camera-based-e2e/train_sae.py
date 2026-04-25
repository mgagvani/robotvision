import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.sae import SparseAutoencoder, TOPK_AUX_SAE_TYPE
from sae_utils import (
    DRVLA_SAE_VERSION,
    DRVLA_SOURCE_NOTE,
    DRVLA_SOURCE_URLS,
    parse_blocks,
    resolve_token_tensor,
    resolve_topk,
)


def compute_centered_mse(x: torch.Tensor) -> float:
    centered = x - x.mean(dim=0, keepdim=True)
    return float(centered.pow(2).mean().clamp_min(1e-8).item())


def approximate_geometric_median(
    x: torch.Tensor,
    *,
    max_samples: int,
    max_iters: int = 100,
    tol: float = 1e-5,
) -> torch.Tensor:
    if x.size(0) > max_samples:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        indices = torch.randperm(x.size(0), generator=generator)[:max_samples]
        x = x[indices]

    median = x.mean(dim=0)
    for _ in range(max_iters):
        distances = torch.norm(x - median.unsqueeze(0), dim=1).clamp_min(1e-8)
        weights = 1.0 / distances
        next_median = (weights[:, None] * x).sum(dim=0) / weights.sum()
        if torch.norm(next_median - median).item() < tol:
            median = next_median
            break
        median = next_median
    return median


def evaluate_epoch(
    model: SparseAutoencoder,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "aux_loss": 0.0,
    }
    total_items = 0

    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            loss_dict = model.compute_loss(batch_x)
            batch_size = batch_x.size(0)
            total_items += batch_size
            for key in totals:
                totals[key] += float(loss_dict[key].item()) * batch_size

    return {key: value / max(total_items, 1) for key, value in totals.items()}


def train_one_block(
    *,
    block_idx: int,
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    lr: float,
    beta1: float,
    beta2: float,
    max_epochs: int,
    k: int,
    k_aux: int,
    lambda_aux: float,
    dead_steps_threshold: int,
    geometric_median_samples: int,
    grad_clip: float,
    seed: int,
    train_dataset_path: str,
    val_dataset_path: str,
    token_key: str,
    train_blob_meta: dict,
    val_blob_meta: dict,
) -> None:
    torch.manual_seed(seed + block_idx)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    b_pre = approximate_geometric_median(
        train_tensor,
        max_samples=geometric_median_samples,
    )
    cmse = compute_centered_mse(train_tensor)

    model = SparseAutoencoder(
        input_dim=train_tensor.shape[1],
        latent_dim=train_tensor.shape[1],
        sae_type=TOPK_AUX_SAE_TYPE,
        k=k,
        k_aux=k_aux,
        aux_alpha=lambda_aux,
        dead_steps_threshold=dead_steps_threshold,
        use_encoder_bias=False,
    ).to(device)
    model.set_preprocessing_state(b_pre=b_pre, cmse=cmse)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    best_score = float("inf")
    best_state = None
    history = []

    for epoch in range(max_epochs):
        model.train()
        running = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "aux_loss": 0.0,
        }
        total_items = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            loss_dict = model.compute_loss(batch_x)

            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            model.project_decoder_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.normalize_decoder_columns()
            model.update_activation_history(loss_dict["z"].detach())

            batch_size_actual = batch_x.size(0)
            total_items += batch_size_actual
            for key in running:
                running[key] += float(loss_dict[key].item()) * batch_size_actual

        train_metrics = {key: value / max(total_items, 1) for key, value in running.items()}
        val_metrics = evaluate_epoch(model, val_loader, device=device)
        dead_fraction = model.dead_fraction()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "train_aux_loss": train_metrics["aux_loss"],
                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_aux_loss": val_metrics["aux_loss"],
                "dead_fraction": dead_fraction,
            }
        )

        print(
            f"[block {block_idx}] epoch {epoch + 1}/{max_epochs} "
            f"train={train_metrics['loss']:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"dead={dead_fraction:.3f}",
            flush=True,
        )

        if val_metrics["loss"] < best_score:
            best_score = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "best_score": best_score,
            "input_dim": train_tensor.shape[1],
            "latent_dim": train_tensor.shape[1],
            "sae_type": TOPK_AUX_SAE_TYPE,
            "block_index": block_idx,
            "token_key": token_key,
            "expansion_ratio": 1.0,
            "k": k,
            "k_aux": k_aux,
            "lambda_aux": lambda_aux,
            "dead_steps_threshold": dead_steps_threshold,
            "lr": lr,
            "betas": (beta1, beta2),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "geometric_median_samples": geometric_median_samples,
            "grad_clip": grad_clip,
            "seed": seed,
            "train_tokens_path": str(Path(train_dataset_path).resolve()),
            "val_tokens_path": str(Path(val_dataset_path).resolve()),
            "cmse": cmse,
            "history": history,
            "metadata": {
                "sae_version": DRVLA_SAE_VERSION,
                "source_note": DRVLA_SOURCE_NOTE,
                "source_urls": list(DRVLA_SOURCE_URLS),
                "activation_kind": "post_transformer_block_query",
                "activation_key": token_key,
                "block_index": block_idx,
                "resolved_hyperparameters": {
                    "input_dim": train_tensor.shape[1],
                    "latent_dim": train_tensor.shape[1],
                    "expansion_ratio": 1.0,
                    "k": k,
                    "k_aux": k_aux,
                    "lambda_aux": lambda_aux,
                    "lr": lr,
                    "beta1": beta1,
                    "beta2": beta2,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "dead_steps_threshold": dead_steps_threshold,
                    "geometric_median_samples": geometric_median_samples,
                    "grad_clip": grad_clip,
                },
                "train_token_meta": train_blob_meta,
                "val_token_meta": val_blob_meta,
            },
        },
        output_dir / "sae_checkpoint.pt",
    )


def parse_requested_k(raw_k: str) -> int | None:
    text = raw_k.strip().lower()
    if text in {"", "auto"}:
        return None
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--blocks", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=str, default="auto")
    parser.add_argument("--k_aux", type=int, default=512)
    parser.add_argument("--lambda_aux", type=float, default=1.0 / 32.0)
    parser.add_argument("--dead_steps_threshold", type=int, default=500)
    parser.add_argument("--geometric_median_samples", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    train_blob = torch.load(args.train_dataset, map_location="cpu")
    val_blob = torch.load(args.val_dataset, map_location="cpu")
    n_blocks = int(train_blob.get("meta", {}).get("n_blocks", 4))
    blocks = parse_blocks(args.blocks, default_blocks=list(range(n_blocks)))
    requested_k = parse_requested_k(args.k)

    for block_idx in blocks:
        train_tensor, token_key = resolve_token_tensor(train_blob, block_idx)
        val_tensor, _ = resolve_token_tensor(val_blob, block_idx)
        k = resolve_topk(train_tensor.shape[1], requested_k=requested_k)
        block_output_dir = output_dir / f"block_{block_idx}"
        print(
            f"Training SAE for block {block_idx} using {token_key}: "
            f"train={tuple(train_tensor.shape)} val={tuple(val_tensor.shape)} k={k}",
            flush=True,
        )
        train_one_block(
            block_idx=block_idx,
            train_tensor=train_tensor,
            val_tensor=val_tensor,
            output_dir=block_output_dir,
            device=device,
            batch_size=args.batch_size,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            max_epochs=args.max_epochs,
            k=k,
            k_aux=args.k_aux,
            lambda_aux=args.lambda_aux,
            dead_steps_threshold=args.dead_steps_threshold,
            geometric_median_samples=args.geometric_median_samples,
            grad_clip=args.grad_clip,
            seed=args.seed,
            train_dataset_path=args.train_dataset,
            val_dataset_path=args.val_dataset,
            token_key=token_key,
            train_blob_meta=dict(train_blob.get("meta", {})),
            val_blob_meta=dict(val_blob.get("meta", {})),
        )


if __name__ == "__main__":
    main()
