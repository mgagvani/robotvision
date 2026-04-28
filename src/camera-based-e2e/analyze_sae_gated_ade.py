import argparse
import csv
from pathlib import Path

import torch

from extract_planner_tok import load_model
from sae_utils import (
    DEFAULT_SAE_BLOCK,
    build_sae_from_checkpoint,
    collate_dataset_indices,
    dataset_from_token_blob,
    default_device,
    load_sae_bundle,
    prepare_replay_context,
    resolve_token_tensor,
)


PROXY_RULE_SPECS = (
    ("selected_score_down", "selected_score", -1.0),
    ("score_margin_up", "score_margin", 1.0),
    ("score_entropy_down", "score_entropy", -1.0),
    ("proposal_spread_down", "proposal_spread", -1.0),
    ("proposal_spread_up", "proposal_spread", 1.0),
)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


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


def compute_output_stats(
    trajectory_flat: torch.Tensor,
    scores: torch.Tensor,
    num_proposals: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    batch = scores.size(0)
    traj = trajectory_flat.view(batch, num_proposals, horizon, 2)
    row_idx = torch.arange(batch, device=scores.device)
    selected_idx = scores.argmin(dim=1)
    selected_score = scores[row_idx, selected_idx]

    if scores.size(1) > 1:
        sorted_scores = scores.sort(dim=1).values
        score_margin = sorted_scores[:, 1] - sorted_scores[:, 0]
    else:
        score_margin = torch.zeros(batch, device=scores.device, dtype=scores.dtype)

    score_probs = torch.softmax(-scores, dim=1)
    score_entropy = -(score_probs * score_probs.clamp_min(1e-8).log()).sum(dim=1)

    traj_mean = traj.mean(dim=1, keepdim=True)
    proposal_spread = torch.norm(traj - traj_mean, dim=-1).mean(dim=(1, 2))

    return {
        "selected_score": selected_score,
        "score_margin": score_margin,
        "score_entropy": score_entropy,
        "proposal_spread": proposal_spread,
    }


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


def compute_proxy_threshold_specs(
    positive_values: torch.Tensor,
    quantiles: list[float],
) -> list[dict]:
    specs = [{"proxy_threshold_name": "gt_zero", "proxy_threshold_value": 0.0}]
    if positive_values.numel() == 0:
        return specs

    seen = {"gt_zero"}
    for q in quantiles:
        threshold_value = float(torch.quantile(positive_values, q).item())
        threshold_name = f"pos_q{int(round(q * 100)):02d}"
        if threshold_name in seen:
            continue
        specs.append(
            {
                "proxy_threshold_name": threshold_name,
                "proxy_threshold_value": threshold_value,
            }
        )
        seen.add(threshold_name)
    return specs


def parse_feature_spec(text: str, latent_dim: int) -> list[int]:
    raw = text.strip().lower()
    if raw in {"all", "*"}:
        return list(range(latent_dim))

    if ":" in raw and "," not in raw:
        parts = raw.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(f"Unsupported feature slice: {text}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else latent_dim
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, stop, step))

    return parse_int_list(text)


def clip_feature_list(
    features: list[int],
    latent_dim: int,
    feature_start: int | None,
    feature_end: int | None,
) -> list[int]:
    clipped = []
    for feature_idx in features:
        if feature_idx < 0 or feature_idx >= latent_dim:
            continue
        if feature_start is not None and feature_idx < feature_start:
            continue
        if feature_end is not None and feature_idx >= feature_end:
            continue
        clipped.append(feature_idx)
    return clipped


def safe_mean(value_sum: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return value_sum / count


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def derive_default_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")


def build_proxy_row(
    *,
    feature_idx: int,
    active_count: int,
    alpha: float,
    scale: float,
    threshold_name: str,
    threshold_value: float,
    gate_rate: float,
    intervened_count: int,
    total_count: int,
    delta_selected: torch.Tensor,
    delta_oracle: torch.Tensor,
    accept_mask: torch.Tensor,
    proxy_rule: str,
    proxy_threshold_name: str,
    proxy_threshold_value: float,
) -> dict:
    accepted_count = int(accept_mask.sum().item())
    if accepted_count > 0:
        delta_selected_accept = delta_selected[accept_mask]
        delta_oracle_accept = delta_oracle[accept_mask]
        improved_selected_accept = int((delta_selected_accept < 0).sum().item())
        improved_oracle_accept = int((delta_oracle_accept < 0).sum().item())
        selected_sum = float(delta_selected_accept.sum().item())
        oracle_sum = float(delta_oracle_accept.sum().item())
    else:
        improved_selected_accept = 0
        improved_oracle_accept = 0
        selected_sum = 0.0
        oracle_sum = 0.0

    return {
        "feature_idx": feature_idx,
        "active_count": active_count,
        "alpha": alpha,
        "intervention_scale": scale,
        "threshold_name": threshold_name,
        "threshold_value": threshold_value,
        "gate_rate": gate_rate,
        "intervened_scene_count": intervened_count,
        "proxy_rule": proxy_rule,
        "proxy_threshold_name": proxy_threshold_name,
        "proxy_threshold_value": proxy_threshold_value,
        "accepted_scene_count": accepted_count,
        "accepted_rate_all": accepted_count / total_count,
        "accepted_rate_intervened": accepted_count / intervened_count if intervened_count > 0 else 0.0,
        "mean_delta_selected_ade_accepted": safe_mean(selected_sum, accepted_count),
        "frac_improved_selected_ade_accepted": (
            improved_selected_accept / accepted_count if accepted_count > 0 else 0.0
        ),
        "mean_delta_oracle_ade_accepted": safe_mean(oracle_sum, accepted_count),
        "frac_improved_oracle_ade_accepted": (
            improved_oracle_accept / accepted_count if accepted_count > 0 else 0.0
        ),
        "mean_delta_selected_ade_if_applied": selected_sum / total_count,
        "frac_improved_selected_ade_if_applied": improved_selected_accept / total_count,
    }


def choose_best_proxy_row(proxy_rows: list[dict], min_accept_count: int) -> dict:
    eligible = [row for row in proxy_rows if row["accepted_scene_count"] >= min_accept_count]
    candidates = eligible if eligible else proxy_rows
    return min(
        candidates,
        key=lambda row: (
            row["mean_delta_selected_ade_accepted"] if row["accepted_scene_count"] > 0 else float("inf"),
            row["mean_delta_selected_ade_if_applied"],
            -row["accepted_scene_count"],
        ),
    )


def summarize_best_rows(rows: list[dict]) -> list[dict]:
    by_feature: dict[int, list[dict]] = {}
    for row in rows:
        by_feature.setdefault(row["feature_idx"], []).append(row)

    summary_rows = []
    for feature_idx, feature_rows in by_feature.items():
        best_row = min(
            feature_rows,
            key=lambda row: (
                row["best_proxy_mean_delta_selected_ade_accepted"]
                if row["best_proxy_accept_count"] > 0
                else float("inf"),
                row["best_proxy_mean_delta_selected_ade_if_applied"],
                row["mean_delta_selected_ade_intervened"],
            ),
        )
        summary_rows.append(best_row)

    summary_rows.sort(
        key=lambda row: (
            row["best_proxy_mean_delta_selected_ade_accepted"]
            if row["best_proxy_accept_count"] > 0
            else float("inf"),
            row["best_proxy_mean_delta_selected_ade_if_applied"],
        )
    )
    return summary_rows


def evaluate_setting(
    *,
    feature_idx: int,
    active_count: int,
    alpha: float,
    scale: float,
    threshold_name: str,
    threshold_value: float,
    intervene_idx: torch.Tensor,
    token_tensor_cpu: torch.Tensor,
    z_all_cpu: torch.Tensor,
    past_cpu: torch.Tensor,
    future_cpu: torch.Tensor,
    baseline_selected: torch.Tensor,
    baseline_oracle: torch.Tensor,
    baseline_stats_cpu: dict[str, torch.Tensor],
    hard_mask_all: torch.Tensor,
    sae,
    planner_model,
    lit_model,
    sae_block: int,
    dataset,
    batch_size: int,
    device: torch.device,
    proxy_metric_quantiles: list[float],
    min_proxy_accept_count: int,
) -> tuple[dict, list[dict]]:
    total_count = baseline_selected.numel()
    intervene_idx = intervene_idx.to(torch.long)
    intervened_count = int(intervene_idx.numel())
    gate_rate = intervened_count / total_count

    if intervened_count == 0:
        proxy_rows = [
            build_proxy_row(
                feature_idx=feature_idx,
                active_count=active_count,
                alpha=alpha,
                scale=scale,
                threshold_name=threshold_name,
                threshold_value=threshold_value,
                gate_rate=gate_rate,
                intervened_count=0,
                total_count=total_count,
                delta_selected=torch.zeros(0),
                delta_oracle=torch.zeros(0),
                accept_mask=torch.zeros(0, dtype=torch.bool),
                proxy_rule="keep_all_intervened",
                proxy_threshold_name="all",
                proxy_threshold_value=float("-inf"),
            )
        ]
        best_proxy = proxy_rows[0]
        row = {
            "feature_idx": feature_idx,
            "active_count": active_count,
            "alpha": alpha,
            "intervention_scale": scale,
            "threshold_name": threshold_name,
            "threshold_value": threshold_value,
            "gate_rate": gate_rate,
            "mean_delta_selected_ade": 0.0,
            "frac_improved_selected_ade": 0.0,
            "mean_delta_oracle_ade": 0.0,
            "frac_improved_oracle_ade": 0.0,
            "intervened_scene_count": 0,
            "mean_delta_selected_ade_intervened": 0.0,
            "frac_improved_selected_ade_intervened": 0.0,
            "intervened_hard_scene_count": 0,
            "mean_delta_selected_ade_hard_intervened": 0.0,
            "frac_improved_selected_ade_hard_intervened": 0.0,
            "best_proxy_rule": best_proxy["proxy_rule"],
            "best_proxy_threshold_name": best_proxy["proxy_threshold_name"],
            "best_proxy_threshold_value": best_proxy["proxy_threshold_value"],
            "best_proxy_accept_count": best_proxy["accepted_scene_count"],
            "best_proxy_accept_rate_all": best_proxy["accepted_rate_all"],
            "best_proxy_accept_rate_intervened": best_proxy["accepted_rate_intervened"],
            "best_proxy_mean_delta_selected_ade_accepted": best_proxy["mean_delta_selected_ade_accepted"],
            "best_proxy_frac_improved_selected_ade_accepted": best_proxy["frac_improved_selected_ade_accepted"],
            "best_proxy_mean_delta_selected_ade_if_applied": best_proxy["mean_delta_selected_ade_if_applied"],
            "best_proxy_frac_improved_selected_ade_if_applied": best_proxy["frac_improved_selected_ade_if_applied"],
        }
        return row, proxy_rows

    sum_delta_selected = 0.0
    sum_delta_oracle = 0.0
    improved_selected = 0
    improved_oracle = 0
    improved_selected_hard = 0
    intervened_hard_count = 0
    sum_delta_selected_hard = 0.0

    delta_selected_chunks = []
    delta_oracle_chunks = []
    directional_value_chunks = {name: [] for name, _, _ in PROXY_RULE_SPECS}

    sae.eval()
    planner_model.eval()
    with torch.no_grad():
        for start in range(0, intervened_count, batch_size):
            batch_indices = intervene_idx[start : start + batch_size]
            batch_future = future_cpu[batch_indices].to(device)
            batch_baseline_selected = baseline_selected[batch_indices].to(device)
            batch_baseline_oracle = baseline_oracle[batch_indices].to(device)
            batch_hard_mask = hard_mask_all[batch_indices]

            batch_x = token_tensor_cpu[batch_indices].to(device)
            z_batch = z_all_cpu[batch_indices].to(device)
            act_batch = z_batch[:, feature_idx]
            z_batch[:, feature_idx] = (act_batch + alpha * scale).clamp_min(0.0)

            recon_query = sae.decode_to_input(z_batch, reference_x=batch_x)
            if sae_block == DEFAULT_SAE_BLOCK:
                batch_past = past_cpu[batch_indices].to(device)
                out = planner_model.forward_from_planner_query_tok(recon_query, batch_past)
            else:
                batch = collate_dataset_indices(dataset, batch_indices)
                replay_context = prepare_replay_context(planner_model, lit_model, batch, device=device)
                out = planner_model.forward_from_block_query_tok(
                    recon_query,
                    replay_context["past"],
                    replay_context["tokens"],
                    start_block=sae_block,
                )

            selected_ade, oracle_ade = compute_selected_and_oracle_ade(
                trajectory_flat=out["trajectory"],
                scores=out["scores"],
                future=batch_future,
                num_proposals=planner_model.n_proposals,
                horizon=planner_model.horizon,
            )
            out_stats = compute_output_stats(
                trajectory_flat=out["trajectory"],
                scores=out["scores"],
                num_proposals=planner_model.n_proposals,
                horizon=planner_model.horizon,
            )

            delta_selected = (selected_ade - batch_baseline_selected).detach().cpu()
            delta_oracle = (oracle_ade - batch_baseline_oracle).detach().cpu()

            sum_delta_selected += float(delta_selected.sum().item())
            sum_delta_oracle += float(delta_oracle.sum().item())
            improved_selected += int((delta_selected < 0).sum().item())
            improved_oracle += int((delta_oracle < 0).sum().item())

            hard_mask_cpu = batch_hard_mask.cpu()
            intervened_hard_count += int(hard_mask_cpu.sum().item())
            if hard_mask_cpu.any():
                hard_delta = delta_selected[hard_mask_cpu]
                sum_delta_selected_hard += float(hard_delta.sum().item())
                improved_selected_hard += int((hard_delta < 0).sum().item())

            delta_selected_chunks.append(delta_selected)
            delta_oracle_chunks.append(delta_oracle)

            for proxy_name, stat_name, sign in PROXY_RULE_SPECS:
                baseline_stat = baseline_stats_cpu[stat_name][batch_indices].to(device)
                directional_values = sign * (out_stats[stat_name] - baseline_stat)
                directional_value_chunks[proxy_name].append(directional_values.detach().cpu())

    delta_selected_all = torch.cat(delta_selected_chunks, dim=0)
    delta_oracle_all = torch.cat(delta_oracle_chunks, dim=0)

    proxy_rows = [
        build_proxy_row(
            feature_idx=feature_idx,
            active_count=active_count,
            alpha=alpha,
            scale=scale,
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            gate_rate=gate_rate,
            intervened_count=intervened_count,
            total_count=total_count,
            delta_selected=delta_selected_all,
            delta_oracle=delta_oracle_all,
            accept_mask=torch.ones(intervened_count, dtype=torch.bool),
            proxy_rule="keep_all_intervened",
            proxy_threshold_name="all",
            proxy_threshold_value=float("-inf"),
        )
    ]

    for proxy_name, _, _ in PROXY_RULE_SPECS:
        directional_values = torch.cat(directional_value_chunks[proxy_name], dim=0)
        positive_values = directional_values[directional_values > 0]
        for spec in compute_proxy_threshold_specs(positive_values, proxy_metric_quantiles):
            accept_mask = directional_values > spec["proxy_threshold_value"]
            proxy_rows.append(
                build_proxy_row(
                    feature_idx=feature_idx,
                    active_count=active_count,
                    alpha=alpha,
                    scale=scale,
                    threshold_name=threshold_name,
                    threshold_value=threshold_value,
                    gate_rate=gate_rate,
                    intervened_count=intervened_count,
                    total_count=total_count,
                    delta_selected=delta_selected_all,
                    delta_oracle=delta_oracle_all,
                    accept_mask=accept_mask,
                    proxy_rule=proxy_name,
                    proxy_threshold_name=spec["proxy_threshold_name"],
                    proxy_threshold_value=spec["proxy_threshold_value"],
                )
            )

    best_proxy = choose_best_proxy_row(proxy_rows, min_accept_count=min_proxy_accept_count)

    row = {
        "feature_idx": feature_idx,
        "active_count": active_count,
        "alpha": alpha,
        "intervention_scale": scale,
        "threshold_name": threshold_name,
        "threshold_value": threshold_value,
        "gate_rate": gate_rate,
        "mean_delta_selected_ade": sum_delta_selected / total_count,
        "frac_improved_selected_ade": improved_selected / total_count,
        "mean_delta_oracle_ade": sum_delta_oracle / total_count,
        "frac_improved_oracle_ade": improved_oracle / total_count,
        "intervened_scene_count": intervened_count,
        "mean_delta_selected_ade_intervened": safe_mean(sum_delta_selected, intervened_count),
        "frac_improved_selected_ade_intervened": (
            improved_selected / intervened_count if intervened_count > 0 else 0.0
        ),
        "intervened_hard_scene_count": intervened_hard_count,
        "mean_delta_selected_ade_hard_intervened": safe_mean(sum_delta_selected_hard, intervened_hard_count),
        "frac_improved_selected_ade_hard_intervened": (
            improved_selected_hard / intervened_hard_count if intervened_hard_count > 0 else 0.0
        ),
        "best_proxy_rule": best_proxy["proxy_rule"],
        "best_proxy_threshold_name": best_proxy["proxy_threshold_name"],
        "best_proxy_threshold_value": best_proxy["proxy_threshold_value"],
        "best_proxy_accept_count": best_proxy["accepted_scene_count"],
        "best_proxy_accept_rate_all": best_proxy["accepted_rate_all"],
        "best_proxy_accept_rate_intervened": best_proxy["accepted_rate_intervened"],
        "best_proxy_mean_delta_selected_ade_accepted": best_proxy["mean_delta_selected_ade_accepted"],
        "best_proxy_frac_improved_selected_ade_accepted": best_proxy["frac_improved_selected_ade_accepted"],
        "best_proxy_mean_delta_selected_ade_if_applied": best_proxy["mean_delta_selected_ade_if_applied"],
        "best_proxy_frac_improved_selected_ade_if_applied": best_proxy["frac_improved_selected_ade_if_applied"],
    }
    return row, proxy_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--planner_checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--features", type=str, default="all")
    parser.add_argument("--sae_block", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--feature_start", type=int, default=None)
    parser.add_argument("--feature_end", type=int, default=None)
    parser.add_argument("--alphas", type=str, default="1.0,2.0")
    parser.add_argument("--threshold_quantiles", type=str, default="0.5,0.75,0.9,0.95")
    parser.add_argument("--proxy_metric_quantiles", type=str, default="0.75,0.9")
    parser.add_argument("--min_proxy_accept_count", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_proxy_csv", type=str, default=None)
    parser.add_argument("--output_best_csv", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    run_root = Path(args.run_root)
    output_csv = Path(args.output_csv)
    output_proxy_csv = Path(args.output_proxy_csv) if args.output_proxy_csv else None
    output_best_csv = (
        Path(args.output_best_csv)
        if args.output_best_csv
        else derive_default_path(output_csv, "_best_by_feature")
    )

    alphas = parse_float_list(args.alphas)
    threshold_quantiles = parse_float_list(args.threshold_quantiles)
    proxy_metric_quantiles = parse_float_list(args.proxy_metric_quantiles)

    bundle = load_sae_bundle(run_root, args.split, args.sae_block, map_location="cpu")
    sae_ckpt = bundle["ckpt"]
    token_blob = bundle["token_blob"]
    token_tensor, token_key = resolve_token_tensor(token_blob, args.sae_block)

    sae = build_sae_from_checkpoint(sae_ckpt, bundle["legacy_norm"])
    sae.to(device)
    sae.eval()

    planner_model, lit_model = load_model(args.planner_checkpoint, device=device)
    planner_model.eval()

    latent_dim = sae_ckpt["latent_dim"]
    features = clip_feature_list(
        parse_feature_spec(args.features, latent_dim=latent_dim),
        latent_dim=latent_dim,
        feature_start=args.feature_start,
        feature_end=args.feature_end,
    )

    past_cpu = token_blob["past"].float()
    future_cpu = token_blob["future"].float()
    scores_cpu = token_blob["scores"].float()
    trajectory_cpu = token_blob["trajectory"].float()

    baseline_selected, baseline_oracle = compute_selected_and_oracle_ade(
        trajectory_flat=trajectory_cpu,
        scores=scores_cpu,
        future=future_cpu,
        num_proposals=planner_model.n_proposals,
        horizon=planner_model.horizon,
    )
    baseline_stats_cpu = compute_output_stats(
        trajectory_flat=trajectory_cpu,
        scores=scores_cpu,
        num_proposals=planner_model.n_proposals,
        horizon=planner_model.horizon,
    )
    hard_threshold = torch.quantile(baseline_selected, 0.75)
    hard_mask_all = baseline_selected >= hard_threshold

    z_all_cpu = []
    with torch.no_grad():
        for start in range(0, len(token_tensor), args.batch_size):
            batch_x = token_tensor[start : start + args.batch_size].to(device)
            z_all_cpu.append(sae.encode(batch_x).cpu())
    z_all_cpu = torch.cat(z_all_cpu, dim=0)

    active_mask = z_all_cpu > 0
    active_count = active_mask.sum(dim=0)
    active_sum = z_all_cpu.sum(dim=0)
    active_sum_sq = (z_all_cpu * z_all_cpu).sum(dim=0)
    active_mean = active_sum / active_count.clamp_min(1)
    active_var = active_sum_sq / active_count.clamp_min(1) - active_mean.square()
    active_std = torch.sqrt(active_var.clamp_min(0.0))
    scales = torch.maximum(active_std, 0.25 * active_mean).clamp_min(0.05)

    dataset = None
    if args.sae_block != DEFAULT_SAE_BLOCK:
        dataset = dataset_from_token_blob(
            token_blob,
            data_dir=args.data_dir,
            index_file=args.index_file,
        )

    all_indices = torch.arange(len(token_tensor))
    rows = []
    proxy_rows = [] if output_proxy_csv is not None else None

    print(
        f"Running gated ADE analysis for {len(features)} features "
        f"(feature_start={args.feature_start}, feature_end={args.feature_end}, token_key={token_key})",
        flush=True,
    )

    for order_idx, feature_idx in enumerate(features, start=1):
        feature_act = z_all_cpu[:, feature_idx]
        threshold_specs = compute_threshold_specs(feature_act, threshold_quantiles)
        active_feature_count = int(active_count[feature_idx].item())
        scale = float(scales[feature_idx].item())
        print(
            f"[{order_idx}/{len(features)}] feature={feature_idx} active={active_feature_count} scale={scale:.4f}",
            flush=True,
        )

        for alpha in alphas:
            for spec in threshold_specs:
                if spec["threshold_name"] == "always_on":
                    intervene_idx = all_indices
                else:
                    intervene_idx = torch.nonzero(
                        feature_act > spec["threshold_value"],
                        as_tuple=False,
                    ).squeeze(1)

                row, setting_proxy_rows = evaluate_setting(
                    feature_idx=feature_idx,
                    active_count=active_feature_count,
                    alpha=alpha,
                    scale=scale,
                    threshold_name=spec["threshold_name"],
                    threshold_value=spec["threshold_value"],
                    intervene_idx=intervene_idx,
                    token_tensor_cpu=token_tensor,
                    z_all_cpu=z_all_cpu,
                    past_cpu=past_cpu,
                    future_cpu=future_cpu,
                    baseline_selected=baseline_selected,
                    baseline_oracle=baseline_oracle,
                    baseline_stats_cpu=baseline_stats_cpu,
                    hard_mask_all=hard_mask_all,
                    sae=sae,
                    planner_model=planner_model,
                    lit_model=lit_model,
                    sae_block=args.sae_block,
                    dataset=dataset,
                    batch_size=args.batch_size,
                    device=device,
                    proxy_metric_quantiles=proxy_metric_quantiles,
                    min_proxy_accept_count=args.min_proxy_accept_count,
                )
                rows.append(row)
                if proxy_rows is not None:
                    proxy_rows.extend(setting_proxy_rows)

                print(
                    {
                        "feature_idx": feature_idx,
                        "alpha": alpha,
                        "threshold_name": spec["threshold_name"],
                        "gate_rate": row["gate_rate"],
                        "mean_delta_selected_ade_intervened": row["mean_delta_selected_ade_intervened"],
                        "best_proxy_rule": row["best_proxy_rule"],
                        "best_proxy_mean_delta_selected_ade_accepted": row["best_proxy_mean_delta_selected_ade_accepted"],
                        "best_proxy_accept_count": row["best_proxy_accept_count"],
                    },
                    flush=True,
                )

    rows.sort(
        key=lambda row: (
            row["best_proxy_mean_delta_selected_ade_accepted"]
            if row["best_proxy_accept_count"] > 0
            else float("inf"),
            row["best_proxy_mean_delta_selected_ade_if_applied"],
            row["mean_delta_selected_ade_intervened"],
        )
    )
    write_csv(output_csv, rows)

    best_rows = summarize_best_rows(rows)
    write_csv(output_best_csv, best_rows)

    if output_proxy_csv is not None and proxy_rows is not None:
        proxy_rows.sort(
            key=lambda row: (
                row["mean_delta_selected_ade_accepted"]
                if row["accepted_scene_count"] > 0
                else float("inf"),
                row["mean_delta_selected_ade_if_applied"],
                -row["accepted_scene_count"],
            )
        )
        write_csv(output_proxy_csv, proxy_rows)

    print(f"Used token key {token_key}", flush=True)
    print(f"Saved detailed rows to {output_csv}", flush=True)
    print(f"Saved best-per-feature rows to {output_best_csv}", flush=True)
    if output_proxy_csv is not None:
        print(f"Saved proxy-rule rows to {output_proxy_csv}", flush=True)
