import argparse
import csv
from pathlib import Path

import torch

from extract_planner_tok import load_model
from models.sae import SparseAutoencoder


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def compute_selected_and_oracle_ade(
    trajectory_flat: torch.Tensor,
    scores: torch.Tensor,
    future: torch.Tensor,
    num_proposals: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = future.size(0)
    traj = trajectory_flat.view(batch, num_proposals, horizon, 2)
    dist = torch.norm(traj - future[:, None], dim=-1)
    ade_per_mode = dist.mean(dim=-1)
    row_idx = torch.arange(batch, device=future.device)
    selected_idx = scores.argmin(dim=1)
    selected_ade = ade_per_mode[row_idx, selected_idx]
    oracle_ade = ade_per_mode.min(dim=1).values
    return selected_ade, oracle_ade


def compute_threshold_specs(
    feature_act: torch.Tensor,
    quantiles: list[float],
) -> list[dict]:
    specs = [{"threshold_name": "always_on", "threshold_value": float("-inf")}]
    specs.append({"threshold_name": "active_only", "threshold_value": 0.0})

    positive = feature_act[feature_act > 0]
    if positive.numel() == 0:
        return specs

    seen = {spec["threshold_name"] for spec in specs}
    for q in quantiles:
        threshold_value = float(torch.quantile(positive, q).item())
        threshold_name = f"q{int(round(q * 100)):02d}_active"
        if threshold_name in seen:
            continue
        specs.append(
            {
                "threshold_name": threshold_name,
                "threshold_value": threshold_value,
            }
        )
        seen.add(threshold_name)
    return specs


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--planner_checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--alphas", type=str, default="1.0,2.0")
    parser.add_argument("--threshold_quantiles", type=str, default="0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    run_root = Path(args.run_root)
    features = parse_int_list(args.features)
    alphas = parse_float_list(args.alphas)
    threshold_quantiles = parse_float_list(args.threshold_quantiles)

    sae_ckpt = torch.load(run_root / "model" / "sae_checkpoint.pt", map_location="cpu")
    norm = torch.load(run_root / "model" / "sae_normalization.pt", map_location="cpu")
    token_blob = torch.load(run_root / "tokens" / f"planner_tokens_{args.split}.pt", map_location="cpu")

    sae = SparseAutoencoder(
        input_dim=sae_ckpt["input_dim"],
        latent_dim=sae_ckpt["latent_dim"],
    )
    sae.load_state_dict(sae_ckpt["state_dict"])
    sae.to(device)
    sae.eval()

    planner_model, _ = load_model(args.planner_checkpoint, device=device)
    planner_model.eval()

    mean = norm["mean"].to(device)
    std = norm["std"].to(device)

    token_tensor = token_blob["planner_query_tok"].float()
    past_cpu = token_blob["past"].float()
    future_cpu = token_blob["future"].float()
    scores_cpu = token_blob["scores"].float()
    trajectory_cpu = token_blob["trajectory"].float()

    # Compute baseline ADE using the saved planner outputs.
    baseline_selected, baseline_oracle = compute_selected_and_oracle_ade(
        trajectory_flat=trajectory_cpu,
        scores=scores_cpu,
        future=future_cpu,
        num_proposals=planner_model.n_proposals,
        horizon=planner_model.horizon,
    )
    hard_threshold = torch.quantile(baseline_selected, 0.75)
    hard_mask_all = baseline_selected >= hard_threshold

    # Encode once for all validation scenes and keep on CPU for thresholding.
    latent_chunks = []
    with torch.no_grad():
        for start in range(0, len(token_tensor), args.batch_size):
            batch_x = token_tensor[start : start + args.batch_size].to(device)
            latent_chunks.append(sae.encode(normalize(batch_x, mean, std)).cpu())
    z_all_cpu = torch.cat(latent_chunks, dim=0)

    active_mask = z_all_cpu > 0
    active_count = active_mask.sum(dim=0)
    active_sum = z_all_cpu.sum(dim=0)
    active_sum_sq = (z_all_cpu * z_all_cpu).sum(dim=0)
    active_mean = active_sum / active_count.clamp_min(1)
    active_var = active_sum_sq / active_count.clamp_min(1) - active_mean.square()
    active_std = torch.sqrt(active_var.clamp_min(0.0))
    scales = torch.maximum(active_std, 0.25 * active_mean).clamp_min(0.05)

    rows = []
    for feature_idx in features:
        feature_act = z_all_cpu[:, feature_idx]
        threshold_specs = compute_threshold_specs(feature_act, threshold_quantiles)
        active_feature_count = int(active_count[feature_idx].item())
        scale = float(scales[feature_idx].item())

        for alpha in alphas:
            accum = {}
            for spec in threshold_specs:
                accum[(alpha, spec["threshold_name"])] = {
                    "sum_delta_selected": 0.0,
                    "sum_delta_oracle": 0.0,
                    "count_improved_selected": 0,
                    "count_improved_oracle": 0,
                    "count_monotone_selected_proxy": 0,
                    "count_intervened": 0,
                    "sum_delta_selected_intervened": 0.0,
                    "count_improved_selected_intervened": 0,
                    "count_intervened_hard": 0,
                    "sum_delta_selected_hard": 0.0,
                    "count_improved_selected_hard": 0,
                }

            with torch.no_grad():
                for start in range(0, len(token_tensor), args.batch_size):
                    end = min(start + args.batch_size, len(token_tensor))
                    batch_tokens = token_tensor[start:end].to(device)
                    batch_past = past_cpu[start:end].to(device)
                    batch_future = future_cpu[start:end].to(device)
                    batch_baseline_selected = baseline_selected[start:end].to(device)
                    batch_baseline_oracle = baseline_oracle[start:end].to(device)
                    batch_hard_mask = hard_mask_all[start:end].to(device)
                    z_batch = z_all_cpu[start:end].to(device)
                    act_batch = z_batch[:, feature_idx]

                    for spec in threshold_specs:
                        if spec["threshold_name"] == "always_on":
                            intervene_mask = torch.ones_like(act_batch, dtype=torch.bool)
                        else:
                            intervene_mask = act_batch > spec["threshold_value"]

                        z_mod = z_batch.clone()
                        if intervene_mask.any():
                            z_mod[intervene_mask, feature_idx] = (
                                act_batch[intervene_mask] + alpha * scale
                            ).clamp_min(0.0)

                        recon_norm = sae.decode(z_mod)
                        recon_query = denormalize(recon_norm, mean, std)
                        out = planner_model.forward_from_planner_query_tok(recon_query, batch_past)
                        selected_ade, oracle_ade = compute_selected_and_oracle_ade(
                            trajectory_flat=out["trajectory"],
                            scores=out["scores"],
                            future=batch_future,
                            num_proposals=planner_model.n_proposals,
                            horizon=planner_model.horizon,
                        )

                        delta_selected = selected_ade - batch_baseline_selected
                        delta_oracle = oracle_ade - batch_baseline_oracle
                        key = (alpha, spec["threshold_name"])
                        state = accum[key]
                        state["sum_delta_selected"] += float(delta_selected.sum().item())
                        state["sum_delta_oracle"] += float(delta_oracle.sum().item())
                        state["count_improved_selected"] += int((delta_selected < 0).sum().item())
                        state["count_improved_oracle"] += int((delta_oracle < 0).sum().item())
                        state["count_intervened"] += int(intervene_mask.sum().item())
                        state["sum_delta_selected_intervened"] += float(delta_selected[intervene_mask].sum().item())
                        state["count_improved_selected_intervened"] += int((delta_selected[intervene_mask] < 0).sum().item())
                        hard_and_intervened = batch_hard_mask & intervene_mask
                        state["count_intervened_hard"] += int(hard_and_intervened.sum().item())
                        state["sum_delta_selected_hard"] += float(delta_selected[hard_and_intervened].sum().item())
                        state["count_improved_selected_hard"] += int((delta_selected[hard_and_intervened] < 0).sum().item())

            total_count = len(token_tensor)
            for spec in threshold_specs:
                key = (alpha, spec["threshold_name"])
                state = accum[key]
                intervened_count = state["count_intervened"]
                intervened_hard_count = state["count_intervened_hard"]
                rows.append(
                    {
                        "feature_idx": feature_idx,
                        "active_count": active_feature_count,
                        "alpha": alpha,
                        "threshold_name": spec["threshold_name"],
                        "threshold_value": spec["threshold_value"],
                        "gate_rate": intervened_count / total_count,
                        "mean_delta_selected_ade": state["sum_delta_selected"] / total_count,
                        "frac_improved_selected_ade": state["count_improved_selected"] / total_count,
                        "mean_delta_oracle_ade": state["sum_delta_oracle"] / total_count,
                        "frac_improved_oracle_ade": state["count_improved_oracle"] / total_count,
                        "intervened_scene_count": intervened_count,
                        "mean_delta_selected_ade_intervened": (
                            state["sum_delta_selected_intervened"] / intervened_count
                            if intervened_count > 0
                            else 0.0
                        ),
                        "frac_improved_selected_ade_intervened": (
                            state["count_improved_selected_intervened"] / intervened_count
                            if intervened_count > 0
                            else 0.0
                        ),
                        "intervened_hard_scene_count": intervened_hard_count,
                        "mean_delta_selected_ade_hard_intervened": (
                            state["sum_delta_selected_hard"] / intervened_hard_count
                            if intervened_hard_count > 0
                            else 0.0
                        ),
                        "frac_improved_selected_ade_hard_intervened": (
                            state["count_improved_selected_hard"] / intervened_hard_count
                            if intervened_hard_count > 0
                            else 0.0
                        ),
                    }
                )

                print(rows[-1])

    rows.sort(key=lambda row: row["mean_delta_selected_ade"])
    write_csv(Path(args.output_csv), rows)
    print(f"Saved CSV to {args.output_csv}")
