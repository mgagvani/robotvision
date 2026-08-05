from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from extract_planner_tok import load_model
from models.sae import SparseAutoencoder
from sae_utils import (
    DEFAULT_SAE_BLOCK,
    build_sae_from_checkpoint,
    collate_dataset_indices,
    dataset_from_token_blob,
    default_device,
    encode_tensor_batchwise,
    load_sae_bundle,
    prepare_replay_context,
    resolve_token_tensor,
)


METRIC_NAMES = (
    "selected_ade",
    "oracle_ade",
    "regret",
    "selected_score",
    "score_margin",
    "score_entropy",
    "proposal_spread",
    "final_lateral_disp",
    "avg_curvature",
    "brake_mag",
    "accel_mag",
)

INTENT_NAMES = {
    0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def infer_checkpoint_path(token_blob: dict) -> str:
    checkpoint = token_blob.get("meta", {}).get("checkpoint")
    if checkpoint and Path(checkpoint).exists():
        return checkpoint
    raise FileNotFoundError(
        "Could not infer planner checkpoint from token metadata. "
        "Pass --planner_checkpoint explicitly."
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def reshape_trajectory(trajectory: torch.Tensor, num_proposals: int, horizon: int) -> torch.Tensor:
    if trajectory.ndim == 4:
        return trajectory
    return trajectory.view(trajectory.size(0), num_proposals, horizon, 2)


def reshape_controls(controls: torch.Tensor, num_proposals: int, horizon: int) -> torch.Tensor:
    if controls.ndim == 4:
        return controls
    return controls.view(controls.size(0), num_proposals, horizon, 2)


def average_curvature(selected_traj: torch.Tensor) -> torch.Tensor:
    if selected_traj.size(1) < 3:
        return torch.zeros(selected_traj.size(0), device=selected_traj.device, dtype=selected_traj.dtype)

    seg = selected_traj[:, 1:] - selected_traj[:, :-1]
    seg_norm = torch.norm(seg, dim=-1).clamp_min(1e-6)
    heading = torch.atan2(seg[..., 1], seg[..., 0])
    d_heading = torch.atan2(
        torch.sin(heading[:, 1:] - heading[:, :-1]),
        torch.cos(heading[:, 1:] - heading[:, :-1]),
    ).abs()
    ds = 0.5 * (seg_norm[:, 1:] + seg_norm[:, :-1])
    return (d_heading / ds.clamp_min(1e-6)).mean(dim=1)


def compute_output_metrics(
    *,
    trajectory: torch.Tensor,
    scores: torch.Tensor,
    controls: torch.Tensor | None,
    future: torch.Tensor,
    num_proposals: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    traj = reshape_trajectory(trajectory.float(), num_proposals=num_proposals, horizon=horizon)
    scores = scores.float()
    future = future.float()

    batch = future.size(0)
    row_idx = torch.arange(batch, device=future.device)
    selected_idx = scores.argmin(dim=1)
    selected_traj = traj[row_idx, selected_idx]

    dist = torch.norm(traj - future[:, None], dim=-1)
    ade_per_mode = dist.mean(dim=-1)
    selected_ade = ade_per_mode[row_idx, selected_idx]
    oracle_ade = ade_per_mode.min(dim=1).values

    sorted_scores = scores.sort(dim=1).values
    if scores.size(1) > 1:
        score_margin = sorted_scores[:, 1] - sorted_scores[:, 0]
    else:
        score_margin = torch.zeros(batch, device=scores.device, dtype=scores.dtype)

    score_probs = torch.softmax(-scores, dim=1)
    score_entropy = -(score_probs * score_probs.clamp_min(1e-8).log()).sum(dim=1)

    traj_mean = traj.mean(dim=1, keepdim=True)
    proposal_spread = torch.norm(traj - traj_mean, dim=-1).mean(dim=(1, 2))

    metrics = {
        "selected_ade": selected_ade,
        "oracle_ade": oracle_ade,
        "regret": selected_ade - oracle_ade,
        "selected_score": scores[row_idx, selected_idx],
        "score_margin": score_margin,
        "score_entropy": score_entropy,
        "proposal_spread": proposal_spread,
        "final_lateral_disp": selected_traj[:, -1, 1],
        "avg_curvature": average_curvature(selected_traj),
    }

    if controls is None:
        zeros = torch.zeros(batch, device=future.device, dtype=future.dtype)
        metrics["brake_mag"] = zeros
        metrics["accel_mag"] = zeros
        return metrics

    ctrl = reshape_controls(controls.float(), num_proposals=num_proposals, horizon=horizon)
    selected_ctrl = ctrl[row_idx, selected_idx]
    accel = selected_ctrl[..., 0]
    metrics["brake_mag"] = (-accel).clamp_min(0).mean(dim=1)
    metrics["accel_mag"] = accel.clamp_min(0).mean(dim=1)
    return metrics


def metrics_from_token_blob(token_blob: dict, num_proposals: int, horizon: int) -> dict[str, torch.Tensor]:
    return compute_output_metrics(
        trajectory=token_blob["trajectory"],
        scores=token_blob["scores"],
        controls=token_blob.get("controls"),
        future=token_blob["future"],
        num_proposals=num_proposals,
        horizon=horizon,
    )


def metrics_from_model_output(
    out: dict[str, torch.Tensor],
    future: torch.Tensor,
    num_proposals: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    return compute_output_metrics(
        trajectory=out["trajectory"],
        scores=out["scores"],
        controls=out.get("controls"),
        future=future,
        num_proposals=num_proposals,
        horizon=horizon,
    )


def zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def select_target_scenes(metrics: dict[str, torch.Tensor], mode: str, count: int) -> torch.Tensor:
    count = min(count, metrics["selected_ade"].numel())
    if mode == "high_regret":
        score = metrics["regret"]
    elif mode == "high_selected_ade":
        score = metrics["selected_ade"]
    elif mode == "low_score_margin":
        score = -metrics["score_margin"]
    elif mode == "high_curvature":
        score = metrics["avg_curvature"].abs()
    elif mode == "high_brake":
        score = metrics["brake_mag"]
    elif mode == "composite_safety":
        score = (
            zscore(metrics["selected_ade"])
            + zscore(metrics["regret"])
            + zscore(metrics["avg_curvature"].abs())
            + zscore(metrics["brake_mag"])
            - zscore(metrics["score_margin"])
        )
    else:
        raise ValueError(f"Unsupported target mode: {mode}")
    return torch.topk(score, k=count).indices.cpu()


def read_float(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_prior_feature_scores(run_root: Path, block: int, split: str) -> dict[int, float]:
    analysis_dir = run_root / "analysis" / f"block_{block}"
    specs = (
        (analysis_dir / f"sae_error_correlation_block_{block}_{split}.csv", ("best_abs_r", "r_selected_ade", "r_regret")),
        (analysis_dir / f"sae_control_summary_block_{block}_{split}.csv", ("best_control_score",)),
        (analysis_dir / f"sae_intent_correlation_block_{block}_{split}.csv", ("eta_sq", "best_abs_r")),
    )
    scores: dict[int, float] = {}
    for path, columns in specs:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                feature_idx_text = row.get("feature_idx")
                if feature_idx_text is None:
                    continue
                feature_idx = int(feature_idx_text)
                score = max(abs(read_float(row, column)) for column in columns)
                scores[feature_idx] = scores.get(feature_idx, 0.0) + score
    return scores


def compute_feature_scales(z_all: torch.Tensor, min_scale: float) -> torch.Tensor:
    active = z_all > 0
    active_count = active.sum(dim=0)
    active_sum = z_all.sum(dim=0)
    active_sum_sq = (z_all * z_all).sum(dim=0)
    active_mean = active_sum / active_count.clamp_min(1)
    active_var = active_sum_sq / active_count.clamp_min(1) - active_mean.square()
    active_std = torch.sqrt(active_var.clamp_min(0.0))
    return torch.maximum(active_std, 0.25 * active_mean).clamp_min(min_scale)


def select_candidate_features(
    *,
    z_all: torch.Tensor,
    target_indices: torch.Tensor,
    feature_budget: int,
    top_active_per_scene: int,
    min_active_frac: float,
    max_active_frac: float,
    prior_scores: dict[int, float],
    prior_weight: float,
) -> list[int]:
    active_frac = (z_all > 0).float().mean(dim=0)
    keep_mask = (active_frac >= min_active_frac) & (active_frac <= max_active_frac)
    target_z = z_all[target_indices]
    target_active_frac = (target_z > 0).float().mean(dim=0)
    target_mean = target_z.mean(dim=0)

    scores = target_mean * (1.0 + target_active_frac)
    if prior_scores:
        prior = torch.zeros_like(scores)
        for feature_idx, score in prior_scores.items():
            if 0 <= feature_idx < prior.numel():
                prior[feature_idx] = float(score)
        scores = scores + prior_weight * prior
    scores = torch.where(keep_mask, scores, torch.full_like(scores, float("-inf")))

    scene_features: set[int] = set()
    if top_active_per_scene > 0:
        k_scene = min(top_active_per_scene, target_z.size(1))
        _, scene_top = torch.topk(target_z, k=k_scene, dim=1)
        for feature_idx in scene_top.flatten().tolist():
            feature_idx = int(feature_idx)
            if bool(keep_mask[feature_idx].item()):
                scene_features.add(feature_idx)

    k_global = min(max(feature_budget, 1), scores.numel())
    global_features = torch.topk(scores, k=k_global).indices.tolist()
    candidates = {int(feature_idx) for feature_idx in global_features if torch.isfinite(scores[feature_idx])}
    candidates.update(scene_features)
    return sorted(candidates, key=lambda feature_idx: float(scores[feature_idx].item()), reverse=True)[:feature_budget]


def run_from_reconstructed_query(
    *,
    block: int,
    reconstructed_query: torch.Tensor,
    scene_indices: torch.Tensor,
    past_cpu: torch.Tensor,
    dataset,
    planner_model,
    lit_model,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if block == DEFAULT_SAE_BLOCK:
        past = past_cpu[scene_indices].to(device)
        return planner_model.forward_from_planner_query_tok(reconstructed_query, past)

    batch = collate_dataset_indices(dataset, scene_indices.cpu())
    replay_context = prepare_replay_context(planner_model, lit_model, batch, device=device)
    return planner_model.forward_from_block_query_tok(
        reconstructed_query,
        replay_context["past"],
        replay_context["tokens"],
        start_block=block,
    )


def replay_to_next_block_query(
    *,
    block: int,
    reconstructed_query: torch.Tensor,
    scene_indices: torch.Tensor,
    dataset,
    planner_model,
    lit_model,
    device: torch.device,
) -> torch.Tensor:
    if block >= len(planner_model.blocks) - 1:
        raise ValueError("Cannot replay to a next block from the final block.")
    batch = collate_dataset_indices(dataset, scene_indices.cpu())
    replay_context = prepare_replay_context(planner_model, lit_model, batch, device=device)
    query = reconstructed_query
    if query.ndim == 2:
        query = query.unsqueeze(1)
    _, block_outputs = planner_model.forward_transformer_blocks(
        query,
        replay_context["tokens"],
        start_block=block + 1,
        return_block_outputs=True,
    )
    return block_outputs[0].squeeze(1)


def feature_relevant_scenes(
    z_all: torch.Tensor,
    target_indices: torch.Tensor,
    feature_idx: int,
    max_scenes: int,
) -> torch.Tensor:
    target_act = z_all[target_indices, feature_idx]
    active = target_act > 0
    active_indices = target_indices[active.cpu()]
    if active_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    keep = min(max_scenes, active_indices.numel())
    _, order = torch.topk(target_act[active], k=keep)
    return active_indices[order.cpu()].to(torch.long)


def summarize_edge_rows(rows: list[dict], key_fields: tuple[str, ...], value_field: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group in grouped.items():
        values = torch.tensor([float(row[value_field]) for row in group], dtype=torch.float32)
        abs_values = values.abs()
        summary = {field: value for field, value in zip(key_fields, key)}
        summary["scene_count"] = len(group)
        summary[f"mean_{value_field}"] = float(values.mean().item())
        summary[f"mean_abs_{value_field}"] = float(abs_values.mean().item())
        summary[f"max_abs_{value_field}"] = float(abs_values.max().item())
        summary[f"frac_positive_{value_field}"] = float((values > 0).float().mean().item())
        summaries.append(summary)

    summaries.sort(key=lambda row: row[f"mean_abs_{value_field}"], reverse=True)
    return summaries


def build_output_edges(
    *,
    block: int,
    feature_idx: int,
    scene_indices: torch.Tensor,
    alphas: list[float],
    sae: SparseAutoencoder,
    token_tensor: torch.Tensor,
    z_all: torch.Tensor,
    scale: float,
    baseline_metrics: dict[str, torch.Tensor],
    metric_stds: dict[str, float],
    future_cpu: torch.Tensor,
    past_cpu: torch.Tensor,
    dataset,
    planner_model,
    lit_model,
    device: torch.device,
) -> list[dict]:
    if scene_indices.numel() == 0:
        return []

    scene_indices = scene_indices.to(torch.long)
    base_x = token_tensor[scene_indices].to(device)
    base_z = z_all[scene_indices].to(device)
    future = future_cpu[scene_indices].to(device)
    source_activation = base_z[:, feature_idx].detach().cpu()

    with torch.no_grad():
        recon_query = sae.decode_to_input(base_z, reference_x=base_x)
        recon_out = run_from_reconstructed_query(
            block=block,
            reconstructed_query=recon_query,
            scene_indices=scene_indices,
            past_cpu=past_cpu,
            dataset=dataset,
            planner_model=planner_model,
            lit_model=lit_model,
            device=device,
        )
        recon_metrics = metrics_from_model_output(
            recon_out,
            future,
            num_proposals=planner_model.n_proposals,
            horizon=planner_model.horizon,
        )

    rows = []
    for alpha in alphas:
        with torch.no_grad():
            z_mod = base_z.clone()
            z_mod[:, feature_idx] = (z_mod[:, feature_idx] + alpha * scale).clamp_min(0.0)
            edited_query = sae.decode_to_input(z_mod, reference_x=base_x)
            edited_out = run_from_reconstructed_query(
                block=block,
                reconstructed_query=edited_query,
                scene_indices=scene_indices,
                past_cpu=past_cpu,
                dataset=dataset,
                planner_model=planner_model,
                lit_model=lit_model,
                device=device,
            )
            edited_metrics = metrics_from_model_output(
                edited_out,
                future,
                num_proposals=planner_model.n_proposals,
                horizon=planner_model.horizon,
            )

        for local_idx, scene_idx in enumerate(scene_indices.tolist()):
            for metric_name in METRIC_NAMES:
                recon_value = float(recon_metrics[metric_name][local_idx].detach().cpu().item())
                edited_value = float(edited_metrics[metric_name][local_idx].detach().cpu().item())
                original_value = float(baseline_metrics[metric_name][scene_idx].item())
                delta = edited_value - recon_value
                rows.append(
                    {
                        "scene_idx": int(scene_idx),
                        "block": block,
                        "feature_idx": feature_idx,
                        "alpha": alpha,
                        "source_activation": float(source_activation[local_idx].item()),
                        "intervention_scale": scale,
                        "metric_name": metric_name,
                        "original_baseline_value": original_value,
                        "reconstruction_baseline_value": recon_value,
                        "intervened_value": edited_value,
                        "delta": delta,
                        "normalized_delta": delta / metric_stds[metric_name],
                        "reconstruction_delta_vs_original": recon_value - original_value,
                    }
                )
    return rows


def build_cross_block_edges(
    *,
    block: int,
    feature_idx: int,
    scene_indices: torch.Tensor,
    alphas: list[float],
    sae: SparseAutoencoder,
    next_sae: SparseAutoencoder,
    token_tensor: torch.Tensor,
    z_all: torch.Tensor,
    next_z_all: torch.Tensor,
    scale: float,
    top_downstream_features: int,
    dataset,
    planner_model,
    lit_model,
    device: torch.device,
) -> list[dict]:
    if scene_indices.numel() == 0:
        return []

    scene_indices = scene_indices.to(torch.long)
    base_x = token_tensor[scene_indices].to(device)
    base_z = z_all[scene_indices].to(device)
    source_activation = base_z[:, feature_idx].detach().cpu()

    with torch.no_grad():
        recon_query = sae.decode_to_input(base_z, reference_x=base_x)
        next_query_recon = replay_to_next_block_query(
            block=block,
            reconstructed_query=recon_query,
            scene_indices=scene_indices,
            dataset=dataset,
            planner_model=planner_model,
            lit_model=lit_model,
            device=device,
        )
        next_z_recon = next_sae.encode(next_query_recon).detach().cpu()

    rows = []
    for alpha in alphas:
        with torch.no_grad():
            z_mod = base_z.clone()
            z_mod[:, feature_idx] = (z_mod[:, feature_idx] + alpha * scale).clamp_min(0.0)
            edited_query = sae.decode_to_input(z_mod, reference_x=base_x)
            next_query_edited = replay_to_next_block_query(
                block=block,
                reconstructed_query=edited_query,
                scene_indices=scene_indices,
                dataset=dataset,
                planner_model=planner_model,
                lit_model=lit_model,
                device=device,
            )
            next_z_edited = next_sae.encode(next_query_edited).detach().cpu()

        delta = next_z_edited - next_z_recon
        k = min(top_downstream_features, delta.size(1))
        top_vals, top_idx = torch.topk(delta.abs(), k=k, dim=1)
        for local_idx, scene_idx in enumerate(scene_indices.tolist()):
            for rank in range(k):
                target_feature = int(top_idx[local_idx, rank].item())
                rows.append(
                    {
                        "scene_idx": int(scene_idx),
                        "source_block": block,
                        "source_feature_idx": feature_idx,
                        "target_block": block + 1,
                        "target_feature_idx": target_feature,
                        "alpha": alpha,
                        "rank": rank + 1,
                        "source_activation": float(source_activation[local_idx].item()),
                        "target_stored_activation": float(next_z_all[scene_idx, target_feature].item()),
                        "target_reconstruction_baseline_activation": float(next_z_recon[local_idx, target_feature].item()),
                        "target_intervened_activation": float(next_z_edited[local_idx, target_feature].item()),
                        "delta": float(delta[local_idx, target_feature].item()),
                        "abs_delta": float(top_vals[local_idx, rank].item()),
                    }
                )
    return rows


def build_target_scene_rows(
    target_indices: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    token_blob: dict,
) -> list[dict]:
    rows = []
    names = token_blob.get("names")
    intent = token_blob.get("intent")
    for rank, scene_idx in enumerate(target_indices.tolist(), start=1):
        row = {"rank": rank, "scene_idx": int(scene_idx)}
        if names is not None:
            row["name"] = names[scene_idx]
        if intent is not None:
            intent_id = int(intent[scene_idx].item())
            row["intent_id"] = intent_id
            row["intent_name"] = INTENT_NAMES.get(intent_id, str(intent_id))
        for metric_name in METRIC_NAMES:
            row[metric_name] = float(metrics[metric_name][scene_idx].item())
        rows.append(row)
    return rows


def build_node_rows(
    *,
    block: int,
    candidate_features: list[int],
    z_all: torch.Tensor,
    target_indices: torch.Tensor,
    prior_scores: dict[int, float],
) -> list[dict]:
    active = z_all > 0
    target_z = z_all[target_indices]
    rows = []
    for feature_idx in candidate_features:
        feature_target = target_z[:, feature_idx]
        rows.append(
            {
                "block": block,
                "feature_idx": feature_idx,
                "active_frac_all": float(active[:, feature_idx].float().mean().item()),
                "active_frac_target": float((feature_target > 0).float().mean().item()),
                "mean_activation_target": float(feature_target.mean().item()),
                "max_activation_target": float(feature_target.max().item()),
                "prior_score": prior_scores.get(feature_idx, 0.0),
            }
        )
    rows.sort(key=lambda row: (row["block"], -row["mean_activation_target"], row["feature_idx"]))
    return rows


def build_scene_graphs(
    *,
    target_indices: torch.Tensor,
    node_rows: list[dict],
    output_edges: list[dict],
    cross_block_edges: list[dict],
) -> list[dict]:
    nodes_by_scene = {int(scene_idx): set() for scene_idx in target_indices.tolist()}
    for row in output_edges:
        nodes_by_scene[row["scene_idx"]].add((row["block"], row["feature_idx"]))
    for row in cross_block_edges:
        nodes_by_scene[row["scene_idx"]].add((row["source_block"], row["source_feature_idx"]))
        nodes_by_scene[row["scene_idx"]].add((row["target_block"], row["target_feature_idx"]))

    node_lookup = {
        (row["block"], row["feature_idx"]): row
        for row in node_rows
    }
    graphs = []
    for scene_idx in target_indices.tolist():
        scene_idx = int(scene_idx)
        nodes = []
        for block, feature_idx in sorted(nodes_by_scene[scene_idx]):
            meta = node_lookup.get((block, feature_idx), {})
            nodes.append(
                {
                    "id": f"b{block}:f{feature_idx}",
                    "kind": "sae_feature",
                    "block": block,
                    "feature_idx": feature_idx,
                    "active_frac_target": meta.get("active_frac_target"),
                    "prior_score": meta.get("prior_score"),
                }
            )

        graph_output_edges = [
            {
                "source": f"b{row['block']}:f{row['feature_idx']}",
                "target": f"metric:{row['metric_name']}",
                "kind": "feature_to_metric",
                "alpha": row["alpha"],
                "delta": row["delta"],
                "normalized_delta": row["normalized_delta"],
            }
            for row in output_edges
            if row["scene_idx"] == scene_idx
        ]
        graph_cross_edges = [
            {
                "source": f"b{row['source_block']}:f{row['source_feature_idx']}",
                "target": f"b{row['target_block']}:f{row['target_feature_idx']}",
                "kind": "feature_to_feature",
                "alpha": row["alpha"],
                "delta": row["delta"],
                "abs_delta": row["abs_delta"],
            }
            for row in cross_block_edges
            if row["scene_idx"] == scene_idx
        ]
        graphs.append(
            {
                "scene_idx": scene_idx,
                "nodes": nodes,
                "edges": graph_cross_edges + graph_output_edges,
            }
        )
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build SAE-based causal attribution graphs for high-risk DeepMonocular planner scenes."
        )
    )
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--planner_checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--blocks", type=str, default="3")
    parser.add_argument("--target_mode", type=str, default="composite_safety",
                        choices=["composite_safety", "high_regret", "high_selected_ade", "low_score_margin", "high_curvature", "high_brake"])
    parser.add_argument("--target_count", type=int, default=32)
    parser.add_argument("--feature_budget", type=int, default=64)
    parser.add_argument("--top_active_per_scene", type=int, default=8)
    parser.add_argument("--max_scenes_per_feature", type=int, default=16)
    parser.add_argument("--top_downstream_features", type=int, default=8)
    parser.add_argument("--alphas", type=str, default="-1.0,1.0")
    parser.add_argument("--min_active_frac", type=float, default=0.001)
    parser.add_argument("--max_active_frac", type=float, default=0.95)
    parser.add_argument("--prior_weight", type=float, default=1.0)
    parser.add_argument("--min_scale", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "analysis" / "circuit_tracing"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    blocks = parse_int_list(args.blocks)
    alphas = parse_float_list(args.alphas)

    if not blocks:
        raise ValueError("--blocks must contain at least one block index.")
    if any(block < 0 for block in blocks):
        raise ValueError("--blocks cannot contain negative block indices.")

    print("Loading SAE bundles and token tensors...", flush=True)
    bundles = {
        block: load_sae_bundle(run_root, args.split, block, map_location="cpu")
        for block in blocks
    }
    token_blob = bundles[blocks[0]]["token_blob"]
    planner_checkpoint = args.planner_checkpoint or infer_checkpoint_path(token_blob)
    planner_model, lit_model = load_model(planner_checkpoint, device=device)
    planner_model.eval()

    if any(block >= len(planner_model.blocks) for block in blocks):
        raise ValueError(f"Block indices must be in [0, {len(planner_model.blocks) - 1}].")

    baseline_metrics = metrics_from_token_blob(
        token_blob,
        num_proposals=planner_model.n_proposals,
        horizon=planner_model.horizon,
    )
    metric_stds = {
        name: float(values.float().std(unbiased=False).clamp_min(1e-6).item())
        for name, values in baseline_metrics.items()
    }
    target_indices = select_target_scenes(
        baseline_metrics,
        mode=args.target_mode,
        count=args.target_count,
    )
    target_rows = build_target_scene_rows(target_indices, baseline_metrics, token_blob)
    write_csv(output_dir / f"target_scenes_{args.split}_{args.target_mode}.csv", target_rows)

    dataset = None
    if any(block != DEFAULT_SAE_BLOCK for block in blocks):
        dataset = dataset_from_token_blob(
            token_blob,
            data_dir=args.data_dir,
            index_file=args.index_file,
        )

    saes: dict[int, SparseAutoencoder] = {}
    token_tensors: dict[int, torch.Tensor] = {}
    z_by_block: dict[int, torch.Tensor] = {}
    scales_by_block: dict[int, torch.Tensor] = {}
    candidate_by_block: dict[int, list[int]] = {}
    prior_by_block: dict[int, dict[int, float]] = {}
    node_rows = []

    for block in blocks:
        print(f"Encoding block {block} SAE activations...", flush=True)
        bundle = bundles[block]
        sae = build_sae_from_checkpoint(bundle["ckpt"], bundle["legacy_norm"]).to(device)
        sae.eval()
        token_tensor, token_key = resolve_token_tensor(bundle["token_blob"], block)
        z_all = encode_tensor_batchwise(
            sae,
            token_tensor,
            batch_size=args.batch_size,
            device=device,
        )
        prior_scores = load_prior_feature_scores(run_root, block, args.split)
        candidates = select_candidate_features(
            z_all=z_all,
            target_indices=target_indices,
            feature_budget=args.feature_budget,
            top_active_per_scene=args.top_active_per_scene,
            min_active_frac=args.min_active_frac,
            max_active_frac=args.max_active_frac,
            prior_scores=prior_scores,
            prior_weight=args.prior_weight,
        )

        print(
            f"Block {block}: token_key={token_key}, candidates={len(candidates)}",
            flush=True,
        )
        saes[block] = sae
        token_tensors[block] = token_tensor
        z_by_block[block] = z_all
        scales_by_block[block] = compute_feature_scales(z_all, min_scale=args.min_scale)
        candidate_by_block[block] = candidates
        prior_by_block[block] = prior_scores
        node_rows.extend(
            build_node_rows(
                block=block,
                candidate_features=candidates,
                z_all=z_all,
                target_indices=target_indices,
                prior_scores=prior_scores,
            )
        )

    write_csv(output_dir / f"feature_nodes_{args.split}_{args.target_mode}.csv", node_rows)

    output_edges = []
    cross_block_edges = []
    past_cpu = token_blob["past"].float()
    future_cpu = token_blob["future"].float()

    for block in blocks:
        print(f"Tracing output edges for block {block}...", flush=True)
        for order_idx, feature_idx in enumerate(candidate_by_block[block], start=1):
            scene_indices = feature_relevant_scenes(
                z_by_block[block],
                target_indices,
                feature_idx,
                max_scenes=args.max_scenes_per_feature,
            )
            if scene_indices.numel() == 0:
                continue
            output_edges.extend(
                build_output_edges(
                    block=block,
                    feature_idx=feature_idx,
                    scene_indices=scene_indices,
                    alphas=alphas,
                    sae=saes[block],
                    token_tensor=token_tensors[block],
                    z_all=z_by_block[block],
                    scale=float(scales_by_block[block][feature_idx].item()),
                    baseline_metrics=baseline_metrics,
                    metric_stds=metric_stds,
                    future_cpu=future_cpu,
                    past_cpu=past_cpu,
                    dataset=dataset,
                    planner_model=planner_model,
                    lit_model=lit_model,
                    device=device,
                )
            )
            if order_idx % 10 == 0 or order_idx == len(candidate_by_block[block]):
                print(
                    f"  block {block}: processed {order_idx}/{len(candidate_by_block[block])} output features",
                    flush=True,
                )

        next_block = block + 1
        if next_block not in blocks:
            continue
        print(f"Tracing cross-block edges {block} -> {next_block}...", flush=True)
        for order_idx, feature_idx in enumerate(candidate_by_block[block], start=1):
            scene_indices = feature_relevant_scenes(
                z_by_block[block],
                target_indices,
                feature_idx,
                max_scenes=args.max_scenes_per_feature,
            )
            if scene_indices.numel() == 0:
                continue
            cross_block_edges.extend(
                build_cross_block_edges(
                    block=block,
                    feature_idx=feature_idx,
                    scene_indices=scene_indices,
                    alphas=alphas,
                    sae=saes[block],
                    next_sae=saes[next_block],
                    token_tensor=token_tensors[block],
                    z_all=z_by_block[block],
                    next_z_all=z_by_block[next_block],
                    scale=float(scales_by_block[block][feature_idx].item()),
                    top_downstream_features=args.top_downstream_features,
                    dataset=dataset,
                    planner_model=planner_model,
                    lit_model=lit_model,
                    device=device,
                )
            )
            if order_idx % 10 == 0 or order_idx == len(candidate_by_block[block]):
                print(
                    f"  block {block}: processed {order_idx}/{len(candidate_by_block[block])} cross-block features",
                    flush=True,
                )

    output_edge_path = output_dir / f"output_edges_{args.split}_{args.target_mode}.csv"
    cross_edge_path = output_dir / f"cross_block_edges_{args.split}_{args.target_mode}.csv"
    output_summary_path = output_dir / f"output_edge_summary_{args.split}_{args.target_mode}.csv"
    cross_summary_path = output_dir / f"cross_block_edge_summary_{args.split}_{args.target_mode}.csv"
    graph_path = output_dir / f"scene_graphs_{args.split}_{args.target_mode}.jsonl"
    summary_path = output_dir / f"summary_{args.split}_{args.target_mode}.json"

    output_summary = summarize_edge_rows(
        output_edges,
        key_fields=("block", "feature_idx", "alpha", "metric_name"),
        value_field="normalized_delta",
    )
    cross_summary = summarize_edge_rows(
        cross_block_edges,
        key_fields=("source_block", "source_feature_idx", "target_block", "target_feature_idx", "alpha"),
        value_field="delta",
    )
    scene_graphs = build_scene_graphs(
        target_indices=target_indices,
        node_rows=node_rows,
        output_edges=output_edges,
        cross_block_edges=cross_block_edges,
    )

    write_csv(output_edge_path, output_edges)
    write_csv(cross_edge_path, cross_block_edges)
    write_csv(output_summary_path, output_summary)
    write_csv(cross_summary_path, cross_summary)
    write_jsonl(graph_path, scene_graphs)

    summary = {
        "run_root": str(run_root),
        "planner_checkpoint": planner_checkpoint,
        "split": args.split,
        "target_mode": args.target_mode,
        "target_count": int(target_indices.numel()),
        "blocks": blocks,
        "alphas": alphas,
        "feature_budget": args.feature_budget,
        "max_scenes_per_feature": args.max_scenes_per_feature,
        "num_nodes": len(node_rows),
        "num_output_edges": len(output_edges),
        "num_cross_block_edges": len(cross_block_edges),
        "metric_stds": metric_stds,
        "artifacts": {
            "target_scenes": str(output_dir / f"target_scenes_{args.split}_{args.target_mode}.csv"),
            "feature_nodes": str(output_dir / f"feature_nodes_{args.split}_{args.target_mode}.csv"),
            "output_edges": str(output_edge_path),
            "output_edge_summary": str(output_summary_path),
            "cross_block_edges": str(cross_edge_path),
            "cross_block_edge_summary": str(cross_summary_path),
            "scene_graphs": str(graph_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("Done.", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
