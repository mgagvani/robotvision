import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.sae import SparseAutoencoder
from sae_utils import (
    build_sae_from_checkpoint,
    default_analysis_dir,
    default_device,
    load_sae_bundle,
    resolve_token_tensor,
)


INTENT_NAMES = {
    0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}
def pearson_from_sums(
    n: int,
    sum_x: torch.Tensor,
    sum_x2: torch.Tensor,
    sum_y: float,
    sum_y2: float,
    sum_xy: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    n_float = float(n)
    mean_x = sum_x / n_float
    mean_y = sum_y / n_float
    var_x = torch.clamp(sum_x2 / n_float - mean_x.square(), min=0.0)
    var_y = max(sum_y2 / n_float - mean_y * mean_y, 0.0)
    std_x = torch.sqrt(var_x)
    std_y = math.sqrt(var_y)
    cov_xy = sum_xy / n_float - mean_x * mean_y

    r = torch.zeros_like(mean_x)
    if std_y > 0:
        mask = std_x > 0
        r[mask] = cov_xy[mask] / (std_x[mask] * std_y)
    return r, mean_y, std_y


def compute_ade_metrics(token_blob: dict) -> dict:
    future = token_blob["future"].float()  # (N, T, 2)
    trajectory = token_blob["trajectory"].float()  # (N, K*T*2)
    scores = token_blob["scores"].float()  # (N, K)
    intent = token_blob["intent"].long()

    n = future.shape[0]
    t = future.shape[1]
    k = scores.shape[1]

    pred = trajectory.view(n, k, t, 2)
    dist = torch.norm(pred - future[:, None, :, :], dim=-1)
    ade_per_mode = dist.mean(dim=-1)

    selected_idx = scores.argmin(dim=1)
    row_idx = torch.arange(n)
    selected_ade = ade_per_mode[row_idx, selected_idx]
    oracle_ade = ade_per_mode.min(dim=1).values
    regret = selected_ade - oracle_ade

    return {
        "selected_ade": selected_ade,
        "oracle_ade": oracle_ade,
        "regret": regret,
        "intent": intent,
    }


def compute_feature_stats(
    model: SparseAutoencoder,
    token_tensor: torch.Tensor,
    metric_tensors: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict:
    metric_names = list(metric_tensors.keys())
    latent_dim = model.encoder.out_features
    n_samples = token_tensor.shape[0]

    thresholds = {}
    for name, values in metric_tensors.items():
        thresholds[name] = {
            "low": torch.quantile(values, 0.10).item(),
            "high": torch.quantile(values, 0.90).item(),
        }

    dataset = TensorDataset(
        token_tensor,
        *[metric_tensors[name] for name in metric_names],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sum_x = torch.zeros(latent_dim, dtype=torch.float64)
    sum_x2 = torch.zeros(latent_dim, dtype=torch.float64)

    metric_state = {}
    for name in metric_names:
        metric_state[name] = {
            "sum_y": 0.0,
            "sum_y2": 0.0,
            "sum_xy": torch.zeros(latent_dim, dtype=torch.float64),
            "high_count": 0,
            "low_count": 0,
            "high_sum_x": torch.zeros(latent_dim, dtype=torch.float64),
            "low_sum_x": torch.zeros(latent_dim, dtype=torch.float64),
        }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device, non_blocking=True)
            z = model.encode(batch_x).cpu().to(torch.float64)

            sum_x += z.sum(dim=0)
            sum_x2 += (z * z).sum(dim=0)

            for i, name in enumerate(metric_names, start=1):
                y = batch[i].cpu().to(torch.float64)
                state = metric_state[name]
                state["sum_y"] += float(y.sum().item())
                state["sum_y2"] += float((y * y).sum().item())
                state["sum_xy"] += (z * y.unsqueeze(1)).sum(dim=0)

                high_mask = y >= thresholds[name]["high"]
                low_mask = y <= thresholds[name]["low"]
                if high_mask.any():
                    state["high_count"] += int(high_mask.sum().item())
                    state["high_sum_x"] += z[high_mask].sum(dim=0)
                if low_mask.any():
                    state["low_count"] += int(low_mask.sum().item())
                    state["low_sum_x"] += z[low_mask].sum(dim=0)

    out = {"n_samples": n_samples, "thresholds": thresholds, "metrics": {}}
    for name in metric_names:
        state = metric_state[name]
        r, mean_y, std_y = pearson_from_sums(
            n=n_samples,
            sum_x=sum_x,
            sum_x2=sum_x2,
            sum_y=state["sum_y"],
            sum_y2=state["sum_y2"],
            sum_xy=state["sum_xy"],
        )
        high_mean = state["high_sum_x"] / max(state["high_count"], 1)
        low_mean = state["low_sum_x"] / max(state["low_count"], 1)
        out["metrics"][name] = {
            "r": r,
            "mean_y": mean_y,
            "std_y": std_y,
            "high_threshold": thresholds[name]["high"],
            "low_threshold": thresholds[name]["low"],
            "high_count": state["high_count"],
            "low_count": state["low_count"],
            "high_mean": high_mean,
            "low_mean": low_mean,
            "delta_high_low": high_mean - low_mean,
        }
    return out


def write_csv(stats: dict, output_csv: Path) -> None:
    metric_names = list(stats["metrics"].keys())
    latent_dim = len(next(iter(stats["metrics"].values()))["r"])
    rows = []
    for feature_idx in range(latent_dim):
        row = {"feature_idx": feature_idx}
        best_metric = None
        best_abs_r = -1.0
        for metric in metric_names:
            metric_stats = stats["metrics"][metric]
            r_val = float(metric_stats["r"][feature_idx].item())
            delta_val = float(metric_stats["delta_high_low"][feature_idx].item())
            row[f"r_{metric}"] = r_val
            row[f"delta_high_low_{metric}"] = delta_val
            if abs(r_val) > best_abs_r:
                best_abs_r = abs(r_val)
                best_metric = metric
        row["best_abs_r"] = best_abs_r
        row["best_metric"] = best_metric
        rows.append(row)

    rows.sort(key=lambda item: item["r_selected_ade"], reverse=True)
    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_metric_summary(metric_name: str, metric_stats: dict, top_k: int) -> None:
    r = metric_stats["r"]
    delta = metric_stats["delta_high_low"]
    top_pos = torch.argsort(r, descending=True)[:top_k].tolist()
    top_neg = torch.argsort(r, descending=False)[:top_k].tolist()

    print(
        f"{metric_name}: mean={metric_stats['mean_y']:.4f} std={metric_stats['std_y']:.4f} "
        f"low10<={metric_stats['low_threshold']:.4f} high10>={metric_stats['high_threshold']:.4f}"
    )
    print(f"Top {top_k} positive correlations:")
    for rank, feature_idx in enumerate(top_pos, start=1):
        print(
            f"  {rank}. feature={feature_idx} "
            f"r={float(r[feature_idx].item()):+.4f} "
            f"delta_high_low={float(delta[feature_idx].item()):+.4f}"
        )
    print(f"Top {top_k} negative correlations:")
    for rank, feature_idx in enumerate(top_neg, start=1):
        print(
            f"  {rank}. feature={feature_idx} "
            f"r={float(r[feature_idx].item()):+.4f} "
            f"delta_high_low={float(delta[feature_idx].item()):+.4f}"
        )
    print("")


def print_ade_by_intent(metrics: dict) -> None:
    selected_ade = metrics["selected_ade"]
    oracle_ade = metrics["oracle_ade"]
    regret = metrics["regret"]
    intent = metrics["intent"]

    print("ADE by intent:")
    for intent_id in sorted(torch.unique(intent).tolist()):
        mask = intent == intent_id
        name = INTENT_NAMES.get(intent_id, str(intent_id))
        print(
            f"  {name}: n={int(mask.sum().item())} "
            f"selected_ADE={float(selected_ade[mask].mean().item()):.4f} "
            f"oracle_ADE={float(oracle_ade[mask].mean().item()):.4f} "
            f"regret={float(regret[mask].mean().item()):.4f}"
        )
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sae_block", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = default_analysis_dir(run_root, args.sae_block, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    bundle = load_sae_bundle(run_root, args.split, args.sae_block, map_location="cpu")
    ckpt = bundle["ckpt"]
    token_blob = bundle["token_blob"]

    model = build_sae_from_checkpoint(ckpt, bundle["legacy_norm"])
    model.to(device)

    token_tensor, token_key = resolve_token_tensor(token_blob, args.sae_block)
    error_metrics = compute_ade_metrics(token_blob)
    intent = error_metrics.pop("intent")

    stats = compute_feature_stats(
        model=model,
        token_tensor=token_tensor,
        metric_tensors=error_metrics,
        batch_size=args.batch_size,
        device=device,
    )

    print_ade_by_intent({**error_metrics, "intent": intent})
    for metric_name, metric_stats in stats["metrics"].items():
        print_metric_summary(metric_name, metric_stats, top_k=args.top_k)

    output_csv = output_dir / f"sae_error_correlation_block_{args.sae_block}_{args.split}.csv"
    write_csv(stats, output_csv)
    print(f"Used token key {token_key}")
    print(f"Saved CSV to {output_csv}")
