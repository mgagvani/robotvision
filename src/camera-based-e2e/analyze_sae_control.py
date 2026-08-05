import argparse
import csv
from pathlib import Path

import torch

from extract_planner_tok import load_model
from models.sae import SparseAutoencoder
from sae_utils import (
    DEFAULT_SAE_BLOCK,
    build_sae_from_checkpoint,
    collate_dataset_indices,
    dataset_from_token_blob,
    default_analysis_dir,
    default_device,
    encode_tensor_batchwise,
    load_sae_bundle,
    planner_inputs_from_collated_batch,
    prepare_replay_context,
    resolve_token_tensor,
)


STAT_NAMES = (
    "final_lateral_disp",
    "avg_curvature",
    "brake_mag",
    "accel_mag",
    "score_margin",
    "proposal_spread",
)


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def infer_checkpoint_path(run_root: Path, token_blob: dict) -> str:
    meta_path = token_blob.get("meta", {}).get("checkpoint")
    if meta_path and Path(meta_path).exists():
        return meta_path

    repo_default = Path(__file__).resolve().parent / "camera-e2e-epoch=04-val_loss=2.90.ckpt"
    if repo_default.exists():
        return str(repo_default)

    raise FileNotFoundError(
        "Could not infer planner checkpoint path. Pass --planner_checkpoint explicitly."
    )


def reshape_trajectory(trajectory: torch.Tensor, num_proposals: int, horizon: int) -> torch.Tensor:
    if trajectory.ndim == 4:
        return trajectory
    return trajectory.view(trajectory.size(0), num_proposals, horizon, 2)


def reshape_controls(controls: torch.Tensor, num_proposals: int, horizon: int) -> torch.Tensor:
    return controls.view(controls.size(0), num_proposals, horizon, 2)


def average_curvature(traj: torch.Tensor) -> torch.Tensor:
    seg = traj[:, 1:] - traj[:, :-1]
    seg_norm = torch.norm(seg, dim=-1).clamp_min(1e-6)
    heading = torch.atan2(seg[..., 1], seg[..., 0])
    d_heading = torch.atan2(
        torch.sin(heading[:, 1:] - heading[:, :-1]),
        torch.cos(heading[:, 1:] - heading[:, :-1]),
    ).abs()
    ds = 0.5 * (seg_norm[:, 1:] + seg_norm[:, :-1])
    curvature = d_heading / ds.clamp_min(1e-6)
    return curvature.mean(dim=1)


def compute_output_stats(
    trajectory: torch.Tensor,
    scores: torch.Tensor,
    controls: torch.Tensor,
    num_proposals: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    traj = reshape_trajectory(trajectory, num_proposals=num_proposals, horizon=horizon)
    ctrl = reshape_controls(controls, num_proposals=num_proposals, horizon=horizon)

    best_idx = scores.argmin(dim=1)
    row_idx = torch.arange(traj.size(0), device=traj.device)

    selected_traj = traj[row_idx, best_idx]
    selected_ctrl = ctrl[row_idx, best_idx]
    accel = selected_ctrl[..., 0]

    sorted_scores = scores.sort(dim=1).values
    if scores.size(1) > 1:
        score_margin = sorted_scores[:, 1] - sorted_scores[:, 0]
    else:
        score_margin = torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)

    traj_mean = traj.mean(dim=1, keepdim=True)
    proposal_spread = torch.norm(traj - traj_mean, dim=-1).mean(dim=(1, 2))

    return {
        "final_lateral_disp": selected_traj[:, -1, 1],
        "avg_curvature": average_curvature(selected_traj),
        "brake_mag": (-accel).clamp_min(0).mean(dim=1),
        "accel_mag": accel.clamp_min(0).mean(dim=1),
        "score_margin": score_margin,
        "proposal_spread": proposal_spread,
    }


def compute_global_stat_stds(
    token_blob: dict,
    num_proposals: int,
    horizon: int,
) -> dict[str, float]:
    stats = compute_output_stats(
        trajectory=token_blob["trajectory"].float(),
        scores=token_blob["scores"].float(),
        controls=token_blob["controls"].float(),
        num_proposals=num_proposals,
        horizon=horizon,
    )
    return {
        name: float(values.std(unbiased=False).clamp_min(1e-6).item())
        for name, values in stats.items()
    }


def rank_along_levels(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float32)
    base = torch.arange(values.size(0), device=values.device, dtype=torch.float32).unsqueeze(1)
    ranks.scatter_(0, order, base.expand_as(order))
    return ranks


def spearman_vs_level(curves: torch.Tensor) -> torch.Tensor:
    if curves.size(0) < 2:
        return torch.zeros(curves.size(1), dtype=torch.float32)
    x = torch.arange(curves.size(0), device=curves.device, dtype=torch.float32)
    x = x - x.mean()
    x_denom = torch.sqrt((x * x).sum()).clamp_min(1e-6)

    y = rank_along_levels(curves)
    y = y - y.mean(dim=0, keepdim=True)
    y_denom = torch.sqrt((y * y).sum(dim=0)).clamp_min(1e-6)
    return (x[:, None] * y).sum(dim=0) / (x_denom * y_denom)


def intervention_levels_for_feature(
    base_activation: torch.Tensor,
    scale: float,
    alphas: list[float],
) -> list[torch.Tensor]:
    return [(base_activation + alpha * scale).clamp_min(0.0) for alpha in alphas]


def run_intervention_outputs(
    *,
    sae_block: int,
    sae: SparseAutoencoder,
    planner_model,
    lit_model,
    token_tensor_cpu: torch.Tensor,
    past_cpu: torch.Tensor,
    relevant_indices: torch.Tensor,
    level_values: list[torch.Tensor],
    base_z_cpu: torch.Tensor,
    dataset,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    base_x = token_tensor_cpu[relevant_indices].to(device)
    base_z = base_z_cpu[relevant_indices].to(device)

    replay_context = None
    if sae_block == DEFAULT_SAE_BLOCK:
        past = past_cpu[relevant_indices].to(device)
    else:
        batch = collate_dataset_indices(dataset, relevant_indices)
        replay_context = prepare_replay_context(planner_model, lit_model, batch, device=device)
        past = replay_context["past"]

    outputs = []
    sae.eval()
    planner_model.eval()
    with torch.no_grad():
        for level in level_values:
            z_mod = base_z.clone()
            feature_idx = None
            # Caller overwrites a single feature before passing level values,
            # so reuse the matching level tensor positionally below.
            if level.ndim != 1:
                raise ValueError("Expected per-scene latent level vector.")
            # Identify the edited feature by comparing shapes later in caller.
            # z_mod is updated in caller before use.
            outputs.append((z_mod, past, base_x, replay_context))
    return outputs


def analyze_feature(
    feature_idx: int,
    token_tensor_cpu: torch.Tensor,
    base_z_cpu: torch.Tensor,
    feature_active_count: int,
    relevant_indices: torch.Tensor,
    scale: float,
    alphas: list[float],
    sae: SparseAutoencoder,
    planner_model,
    lit_model,
    sae_block: int,
    past_cpu: torch.Tensor,
    global_stat_stds: dict[str, float],
    dataset,
    device: torch.device,
) -> tuple[list[dict], dict]:
    if relevant_indices.numel() == 0:
        rows = []
        for stat_name in STAT_NAMES:
            rows.append(
                {
                    "feature_idx": feature_idx,
                    "stat_name": stat_name,
                    "active_count": feature_active_count,
                    "relevant_scene_count": 0,
                    "intervention_scale": scale,
                    "mean_delta_max": 0.0,
                    "std_effect_max": 0.0,
                    "mean_rho": 0.0,
                    "frac_consistent": 0.0,
                    "selectivity_share": 0.0,
                    "target_vs_next_ratio": 0.0,
                    "control_score": 0.0,
                }
            )
        summary = {
            "feature_idx": feature_idx,
            "best_stat": "NONE",
            "best_control_score": 0.0,
            "best_std_effect_max": 0.0,
            "best_mean_rho": 0.0,
            "best_frac_consistent": 0.0,
            "best_selectivity_share": 0.0,
            "active_count": feature_active_count,
            "relevant_scene_count": 0,
            "intervention_scale": scale,
            "runner_up_stat": "NONE",
        }
        return rows, summary

    base_x = token_tensor_cpu[relevant_indices].to(device)
    base_z = base_z_cpu[relevant_indices].to(device)
    base_activation = base_z[:, feature_idx].clone()
    level_values = intervention_levels_for_feature(base_activation, scale=scale, alphas=alphas)

    replay_context = None
    if sae_block == DEFAULT_SAE_BLOCK:
        past = past_cpu[relevant_indices].to(device)
    else:
        batch = collate_dataset_indices(dataset, relevant_indices)
        replay_context = prepare_replay_context(planner_model, lit_model, batch, device=device)
        past = replay_context["past"]

    stat_curves = {name: [] for name in STAT_NAMES}
    sae.eval()
    planner_model.eval()
    with torch.no_grad():
        for level in level_values:
            z_mod = base_z.clone()
            z_mod[:, feature_idx] = level
            recon_query = sae.decode_to_input(z_mod, reference_x=base_x)
            if sae_block == DEFAULT_SAE_BLOCK:
                out = planner_model.forward_from_planner_query_tok(recon_query, past)
            else:
                out = planner_model.forward_from_block_query_tok(
                    recon_query,
                    past,
                    replay_context["tokens"],
                    start_block=sae_block,
                )
            stats = compute_output_stats(
                trajectory=out["trajectory"],
                scores=out["scores"],
                controls=out["controls"],
                num_proposals=planner_model.n_proposals,
                horizon=planner_model.horizon,
            )
            for stat_name in STAT_NAMES:
                stat_curves[stat_name].append(stats[stat_name].cpu())

    rows = []
    std_effects = {}
    metric_cache = {}
    for stat_name in STAT_NAMES:
        curves = torch.stack(stat_curves[stat_name], dim=0)
        delta = curves - curves[0:1]
        delta_max = delta[-1]
        mean_delta_max = float(delta_max.mean().item())
        std_effect_max = mean_delta_max / global_stat_stds[stat_name]
        rho_scene = spearman_vs_level(curves)
        mean_rho = float(rho_scene.mean().item())
        sign = 1.0 if mean_delta_max >= 0 else -1.0
        consistent = ((sign * rho_scene) > 0.5) & ((sign * delta_max) > 0)
        frac_consistent = float(consistent.float().mean().item())
        std_effects[stat_name] = abs(std_effect_max)
        metric_cache[stat_name] = {
            "mean_delta_max": mean_delta_max,
            "std_effect_max": std_effect_max,
            "mean_rho": mean_rho,
            "frac_consistent": frac_consistent,
        }

    effect_sum = sum(std_effects.values()) + 1e-8
    sorted_effects = sorted(std_effects.items(), key=lambda item: item[1], reverse=True)

    for stat_name in STAT_NAMES:
        next_best = max(
            (value for other_name, value in std_effects.items() if other_name != stat_name),
            default=0.0,
        )
        selectivity_share = std_effects[stat_name] / effect_sum
        target_vs_next_ratio = std_effects[stat_name] / (next_best + 1e-8)
        control_score = (
            std_effects[stat_name]
            * abs(metric_cache[stat_name]["mean_rho"])
            * metric_cache[stat_name]["frac_consistent"]
            * selectivity_share
        )
        rows.append(
            {
                "feature_idx": feature_idx,
                "stat_name": stat_name,
                "active_count": feature_active_count,
                "relevant_scene_count": int(relevant_indices.numel()),
                "intervention_scale": scale,
                "mean_delta_max": metric_cache[stat_name]["mean_delta_max"],
                "std_effect_max": metric_cache[stat_name]["std_effect_max"],
                "mean_rho": metric_cache[stat_name]["mean_rho"],
                "frac_consistent": metric_cache[stat_name]["frac_consistent"],
                "selectivity_share": selectivity_share,
                "target_vs_next_ratio": target_vs_next_ratio,
                "control_score": control_score,
            }
        )

    best_row = max(rows, key=lambda row: row["control_score"])
    summary = {
        "feature_idx": feature_idx,
        "best_stat": best_row["stat_name"],
        "best_control_score": best_row["control_score"],
        "best_std_effect_max": best_row["std_effect_max"],
        "best_mean_rho": best_row["mean_rho"],
        "best_frac_consistent": best_row["frac_consistent"],
        "best_selectivity_share": best_row["selectivity_share"],
        "active_count": feature_active_count,
        "relevant_scene_count": int(relevant_indices.numel()),
        "intervention_scale": scale,
        "runner_up_stat": sorted_effects[1][0] if len(sorted_effects) > 1 else "NONE",
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top_features(summary_rows: list[dict], stat_rows: list[dict], top_k: int) -> None:
    print("Top controllable features overall:")
    top_features = sorted(summary_rows, key=lambda row: row["best_control_score"], reverse=True)[:top_k]
    for rank, row in enumerate(top_features, start=1):
        print(
            f"  {rank}. feature={row['feature_idx']} best_stat={row['best_stat']} "
            f"score={row['best_control_score']:.4f} std_effect={row['best_std_effect_max']:+.4f} "
            f"rho={row['best_mean_rho']:+.4f} consistent={row['best_frac_consistent']:.3f} "
            f"selectivity={row['best_selectivity_share']:.3f}"
        )
    print("")

    for stat_name in STAT_NAMES:
        top_rows = sorted(
            (row for row in stat_rows if row["stat_name"] == stat_name),
            key=lambda row: row["control_score"],
            reverse=True,
        )[:top_k]
        print(f"Top features for {stat_name}:")
        for rank, row in enumerate(top_rows, start=1):
            print(
                f"  {rank}. feature={row['feature_idx']} score={row['control_score']:.4f} "
                f"std_effect={row['std_effect_max']:+.4f} rho={row['mean_rho']:+.4f} "
                f"consistent={row['frac_consistent']:.3f} selectivity={row['selectivity_share']:.3f}"
            )
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--planner_checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sae_block", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--encode_batch_size", type=int, default=4096)
    parser.add_argument("--relevant_scenes_per_feature", type=int, default=64)
    parser.add_argument("--alphas", type=str, default="0,0.5,1.0,2.0")
    parser.add_argument("--min_scale", type=float, default=0.05)
    parser.add_argument("--max_features", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = default_analysis_dir(run_root, args.sae_block, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    alphas = parse_float_list(args.alphas)

    bundle = load_sae_bundle(run_root, args.split, args.sae_block, map_location="cpu")
    sae_ckpt = bundle["ckpt"]
    token_blob = bundle["token_blob"]
    token_tensor, token_key = resolve_token_tensor(token_blob, args.sae_block)

    planner_checkpoint = args.planner_checkpoint or infer_checkpoint_path(run_root, token_blob)
    planner_model, lit_model = load_model(planner_checkpoint, device=device)

    global_stat_stds = compute_global_stat_stds(
        token_blob=token_blob,
        num_proposals=planner_model.n_proposals,
        horizon=planner_model.horizon,
    )
    print("Global stat stds:")
    for stat_name in STAT_NAMES:
        print(f"  {stat_name}: {global_stat_stds[stat_name]:.4f}")
    print("")

    sae = build_sae_from_checkpoint(sae_ckpt, bundle["legacy_norm"])
    sae.to(device)
    sae.eval()

    past_cpu = token_blob["past"].float()
    base_z_cpu = encode_tensor_batchwise(
        sae,
        token_tensor,
        batch_size=args.encode_batch_size,
        device=device,
    )

    active_mask = base_z_cpu > 0
    active_count = active_mask.sum(dim=0)
    active_sum = base_z_cpu.sum(dim=0)
    active_sum_sq = (base_z_cpu * base_z_cpu).sum(dim=0)
    active_mean = active_sum / active_count.clamp_min(1)
    active_var = active_sum_sq / active_count.clamp_min(1) - active_mean.square()
    active_std = torch.sqrt(active_var.clamp_min(0))
    intervention_scale = torch.maximum(active_std, 0.25 * active_mean).clamp_min(args.min_scale)

    top_k = min(args.relevant_scenes_per_feature, base_z_cpu.size(0))
    _, top_indices = torch.topk(base_z_cpu, k=top_k, dim=0)

    dataset = None
    if args.sae_block != DEFAULT_SAE_BLOCK:
        dataset = dataset_from_token_blob(
            token_blob,
            data_dir=args.data_dir,
            index_file=args.index_file,
        )

    feature_limit = base_z_cpu.size(1) if args.max_features is None else min(args.max_features, base_z_cpu.size(1))
    stat_rows = []
    summary_rows = []

    for feature_idx in range(feature_limit):
        feature_active = int(active_count[feature_idx].item())
        keep = min(feature_active, top_k)
        relevant_indices = top_indices[:keep, feature_idx] if keep > 0 else torch.empty(0, dtype=torch.long)
        scale = float(intervention_scale[feature_idx].item())

        feature_stat_rows, feature_summary = analyze_feature(
            feature_idx=feature_idx,
            token_tensor_cpu=token_tensor,
            base_z_cpu=base_z_cpu,
            feature_active_count=feature_active,
            relevant_indices=relevant_indices,
            scale=scale,
            alphas=alphas,
            sae=sae,
            planner_model=planner_model,
            lit_model=lit_model,
            sae_block=args.sae_block,
            past_cpu=past_cpu,
            global_stat_stds=global_stat_stds,
            dataset=dataset,
            device=device,
        )
        stat_rows.extend(feature_stat_rows)
        summary_rows.append(feature_summary)

        if (feature_idx + 1) % 100 == 0 or feature_idx + 1 == feature_limit:
            print(f"Processed {feature_idx + 1}/{feature_limit} features")

    summary_csv = output_dir / f"sae_control_summary_block_{args.sae_block}_{args.split}.csv"
    stat_csv = output_dir / f"sae_control_stat_rows_block_{args.sae_block}_{args.split}.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(stat_csv, stat_rows)

    print("")
    print(f"Used token key {token_key}")
    print_top_features(summary_rows, stat_rows, top_k=args.top_k)
    print(f"Saved feature summary to {summary_csv}")
    print(f"Saved per-stat rows to {stat_csv}")
