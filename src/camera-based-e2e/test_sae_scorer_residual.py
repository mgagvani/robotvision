"""
Evaluate whether an SAE-informed residual model improves scorer proposal selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from loader import WaymoE2E
from models.base_model import collate_with_images
from sae_scorer_correction import (
    DEFAULT_SEED,
    DEFAULT_VAL_SPLIT,
    SCORE_METRIC_NAMES,
    checkpoint_value,
    correction_loss,
    default_index_file,
    default_num_workers,
    extract_batch_targets,
    flatten_proposal_batch,
    load_model_and_sae,
    load_residual_model,
    resolve_required_arg,
    score_comparison_metrics,
)


def evaluate(
    residual_model,
    loader,
    model,
    sae,
    device: torch.device,
    *,
    loss_type: str,
) -> dict[str, float]:
    residual_model.eval()

    totals = defaultdict(float)
    n_samples = 0
    n_scores = 0

    with torch.no_grad():
        for batch in loader:
            batch_targets = extract_batch_targets(model, sae, batch, device)
            flat_batch = flatten_proposal_batch(batch_targets)
            residual_pred_flat = residual_model(
                flat_batch["sae_hidden"],
                flat_batch["trajectory"],
                flat_batch["score_pred"],
            )
            residual_loss = correction_loss(
                residual_pred_flat,
                flat_batch["residual_target"],
                loss_type=loss_type,
            )
            corrected_scores = batch_targets["score_pred"] + residual_pred_flat.view_as(batch_targets["score_pred"])

            sample_count = batch_targets["score_pred"].size(0)
            score_count = batch_targets["score_pred"].numel()
            n_samples += sample_count
            n_scores += score_count

            totals["residual_loss"] += residual_loss.item() * score_count
            for key, value in score_comparison_metrics(
                batch_targets["score_pred"],
                corrected_scores,
                batch_targets["true_ade"],
            ).items():
                weight = score_count if key in SCORE_METRIC_NAMES else sample_count
                totals[key] += value.item() * weight

    if n_samples == 0 or n_scores == 0:
        raise RuntimeError("No evaluation batches were processed")

    out = {}
    for key, total in totals.items():
        denom = n_scores if "score_" in key or key == "residual_loss" else n_samples
        out[key] = total / denom

    out["delta_score_mse"] = out["corrected_score_mse"] - out["baseline_score_mse"]
    out["delta_score_mae"] = out["corrected_score_mae"] - out["baseline_score_mae"]
    out["delta_picked_ade"] = out["corrected_picked_ade"] - out["baseline_picked_ade"]
    out["delta_regret"] = out["corrected_regret"] - out["baseline_regret"]
    out["delta_mean_rank"] = out["corrected_mean_rank"] - out["baseline_mean_rank"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--sae_checkpoint_path", type=str, default=None)
    parser.add_argument("--residual_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default=DEFAULT_VAL_SPLIT, choices=["train", "val"])
    parser.add_argument("--n_items", type=int, default=25_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=default_num_workers())
    parser.add_argument("--block_idx", type=int, default=None)
    parser.add_argument("--loss_type", type=str, default=None, choices=["mse", "huber"])
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(DEFAULT_SEED, workers=True)

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    residual_model, residual_checkpoint = load_residual_model(
        args.residual_checkpoint_path,
        device=device,
    )
    model_checkpoint_path = resolve_required_arg(
        args.model_checkpoint_path,
        residual_checkpoint,
        "model_checkpoint_path",
    )
    sae_checkpoint_path = resolve_required_arg(
        args.sae_checkpoint_path,
        residual_checkpoint,
        "sae_checkpoint_path",
    )
    block_idx = int(resolve_required_arg(args.block_idx, residual_checkpoint, "block_idx"))
    loss_type = args.loss_type or checkpoint_value(residual_checkpoint, "loss_type") or "huber"

    loader = DataLoader(
        WaymoE2E(
            indexFile=default_index_file(args.split),
            data_dir=args.data_dir,
            n_items=args.n_items,
            seed=DEFAULT_SEED,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
        shuffle=False,
    )

    model, sae, hook_handle = load_model_and_sae(
        model_checkpoint_path=model_checkpoint_path,
        sae_checkpoint_path=sae_checkpoint_path,
        block_idx=block_idx,
        device=device,
    )

    try:
        metrics = evaluate(
            residual_model,
            loader,
            model,
            sae,
            device,
            loss_type=loss_type,
        )
    finally:
        hook_handle.remove()

    print(f"Residual model checkpoint: {args.residual_checkpoint_path}")
    print(f"Backbone model checkpoint: {model_checkpoint_path}")
    print(f"SAE checkpoint: {sae_checkpoint_path}")
    print(f"Using block index: {block_idx}")
    print(f"Using residual loss type: {loss_type}")
    if "best_metric" in residual_checkpoint:
        print(f"Saved best validation corrected pick ADE: {residual_checkpoint['best_metric']:.6f}")
    print(f"Score MSE: {metrics['baseline_score_mse']:.6f} -> {metrics['corrected_score_mse']:.6f} (delta {metrics['delta_score_mse']:+.6f})")
    print(f"Score MAE: {metrics['baseline_score_mae']:.6f} -> {metrics['corrected_score_mae']:.6f} (delta {metrics['delta_score_mae']:+.6f})")
    print(f"Picked ADE: {metrics['baseline_picked_ade']:.6f} -> {metrics['corrected_picked_ade']:.6f} (delta {metrics['delta_picked_ade']:+.6f})")
    print(f"Oracle ADE: {metrics['baseline_oracle_ade']:.6f}")
    print(f"Regret: {metrics['baseline_regret']:.6f} -> {metrics['corrected_regret']:.6f} (delta {metrics['delta_regret']:+.6f})")
    print(f"Mean rank: {metrics['baseline_mean_rank']:.6f} -> {metrics['corrected_mean_rank']:.6f} (delta {metrics['delta_mean_rank']:+.6f})")
    for topk in (1, 5, 10):
        key = f"top{topk}_acc"
        baseline_key = f"baseline_{key}"
        corrected_key = f"corrected_{key}"
        if baseline_key in metrics:
            delta = metrics[corrected_key] - metrics[baseline_key]
            print(f"{key}: {metrics[baseline_key]:.6f} -> {metrics[corrected_key]:.6f} (delta {delta:+.6f})")
    print(f"Spearman: {metrics['baseline_spearman']:.6f} -> {metrics['corrected_spearman']:.6f}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
