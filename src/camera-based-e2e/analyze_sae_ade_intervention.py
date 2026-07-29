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
    default_device,
    encode_tensor_batchwise,
    load_sae_bundle,
    prepare_replay_context,
    resolve_token_tensor,
)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


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


def rank_along_levels(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float32)
    base = torch.arange(values.size(0), device=values.device, dtype=torch.float32).unsqueeze(1)
    ranks.scatter_(0, order, base.expand_as(order))
    return ranks


def spearman_vs_level(curves: torch.Tensor) -> torch.Tensor:
    if curves.size(0) < 2:
        return torch.zeros(curves.size(1), dtype=torch.float32, device=curves.device)
    x = torch.arange(curves.size(0), device=curves.device, dtype=torch.float32)
    x = x - x.mean()
    x_denom = torch.sqrt((x * x).sum()).clamp_min(1e-6)

    y = rank_along_levels(curves)
    y = y - y.mean(dim=0, keepdim=True)
    y_denom = torch.sqrt((y * y).sum(dim=0)).clamp_min(1e-6)
    return (x[:, None] * y).sum(dim=0) / (x_denom * y_denom)


def selected_and_oracle_ade(
    out: dict,
    future_batch: torch.Tensor,
    num_proposals: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = future_batch.size(0)
    traj = out["trajectory"].view(batch, num_proposals, horizon, 2)
    scores = out["scores"]
    dist = torch.norm(traj - future_batch[:, None], dim=-1)
    ade_per_mode = dist.mean(dim=-1)
    row_idx = torch.arange(batch, device=future_batch.device)
    selected_idx = scores.argmin(dim=1)
    selected_ade = ade_per_mode[row_idx, selected_idx]
    oracle_ade = ade_per_mode.min(dim=1).values
    return selected_ade, oracle_ade


def summarize_feature(
    feature_idx: int,
    token_tensor_cpu: torch.Tensor,
    z_all: torch.Tensor,
    scales: torch.Tensor,
    sae: SparseAutoencoder,
    planner_model,
    lit_model,
    sae_block: int,
    past: torch.Tensor,
    future: torch.Tensor,
    relevant_scenes: int,
    alphas: list[float],
    scene_selection: str,
    random_seed: int,
    dataset,
    device: torch.device,
) -> dict:
    feature_act = z_all[:, feature_idx]
    active_mask = feature_act > 0
    active_count = int(active_mask.sum().item())
    if scene_selection == "relevant":
        keep = min(active_count, relevant_scenes)
    else:
        keep = min(z_all.size(0), relevant_scenes)
    if keep == 0:
        return {
            "feature_idx": feature_idx,
            "active_count": 0,
            "relevant_scene_count": 0,
            "scene_selection": scene_selection,
            "random_seed": random_seed,
            "mean_delta_selected_ade": 0.0,
            "frac_improved_selected_ade": 0.0,
            "frac_monotone_down_selected_ade": 0.0,
            "mean_rho_selected_ade": 0.0,
            "mean_delta_selected_ade_hard": 0.0,
            "frac_improved_selected_ade_hard": 0.0,
            "mean_delta_oracle_ade": 0.0,
            "frac_improved_oracle_ade": 0.0,
        }

    if scene_selection == "relevant":
        rel_idx = torch.topk(feature_act, k=keep).indices
    elif scene_selection == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        rel_idx = torch.randperm(z_all.size(0), generator=generator)[:keep]
    else:
        raise ValueError(f"Unsupported scene_selection={scene_selection}")

    z_rel = z_all[rel_idx].clone().to(device)
    future_rel = future[rel_idx].to(device)
    base_x = token_tensor_cpu[rel_idx].to(device)
    base_activation = z_rel[:, feature_idx].clone()
    scale = float(scales[feature_idx].item())

    if sae_block == DEFAULT_SAE_BLOCK:
        replay_context = None
        past_rel = past[rel_idx].to(device)
    else:
        batch = collate_dataset_indices(dataset, rel_idx)
        replay_context = prepare_replay_context(planner_model, lit_model, batch, device=device)
        past_rel = replay_context["past"]

    selected_curves = []
    oracle_curves = []
    with torch.no_grad():
        for alpha in alphas:
            z_mod = z_rel.clone()
            z_mod[:, feature_idx] = (base_activation + alpha * scale).clamp_min(0.0)
            recon_query = sae.decode_to_input(z_mod, reference_x=base_x)
            if sae_block == DEFAULT_SAE_BLOCK:
                out = planner_model.forward_from_planner_query_tok(recon_query, past_rel)
            else:
                out = planner_model.forward_from_block_query_tok(
                    recon_query,
                    past_rel,
                    replay_context["tokens"],
                    start_block=sae_block,
                )
            selected_ade, oracle_ade = selected_and_oracle_ade(
                out=out,
                future_batch=future_rel,
                num_proposals=planner_model.n_proposals,
                horizon=planner_model.horizon,
            )
            selected_curves.append(selected_ade)
            oracle_curves.append(oracle_ade)

    selected_curves = torch.stack(selected_curves, dim=0)
    oracle_curves = torch.stack(oracle_curves, dim=0)

    base_selected = selected_curves[0]
    final_selected = selected_curves[-1]
    delta_selected = final_selected - base_selected

    base_oracle = oracle_curves[0]
    final_oracle = oracle_curves[-1]
    delta_oracle = final_oracle - base_oracle

    hard_thresh = torch.quantile(base_selected, 0.75)
    hard_mask = base_selected >= hard_thresh

    diffs = selected_curves[1:] - selected_curves[:-1]
    monotone_down = (diffs <= 1e-8).all(dim=0)
    rho_selected = spearman_vs_level(selected_curves)

    return {
        "feature_idx": feature_idx,
        "active_count": active_count,
        "relevant_scene_count": keep,
        "scene_selection": scene_selection,
        "random_seed": random_seed,
        "intervention_scale": scale,
        "mean_delta_selected_ade": float(delta_selected.mean().item()),
        "frac_improved_selected_ade": float((delta_selected < 0).float().mean().item()),
        "frac_monotone_down_selected_ade": float(monotone_down.float().mean().item()),
        "mean_rho_selected_ade": float(rho_selected.mean().item()),
        "mean_delta_selected_ade_hard": float(delta_selected[hard_mask].mean().item()),
        "frac_improved_selected_ade_hard": float((delta_selected[hard_mask] < 0).float().mean().item()),
        "mean_delta_oracle_ade": float(delta_oracle.mean().item()),
        "frac_improved_oracle_ade": float((delta_oracle < 0).float().mean().item()),
    }


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
    parser.add_argument("--sae_block", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--feature_start", type=int, default=None)
    parser.add_argument("--feature_end", type=int, default=None)
    parser.add_argument("--alphas", type=str, default="0,0.5,1.0,2.0")
    parser.add_argument("--relevant_scenes_per_feature", type=int, default=384)
    parser.add_argument("--scene_selection", type=str, default="relevant", choices=["relevant", "random"])
    parser.add_argument("--random_seeds", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    run_root = Path(args.run_root)

    bundle = load_sae_bundle(run_root, args.split, args.sae_block, map_location="cpu")
    sae_ckpt = bundle["ckpt"]
    token_blob = bundle["token_blob"]
    token_tensor, token_key = resolve_token_tensor(token_blob, args.sae_block)

    sae = build_sae_from_checkpoint(sae_ckpt, bundle["legacy_norm"])
    sae.to(device)
    sae.eval()

    planner_model, lit_model = load_model(args.planner_checkpoint, device=device)
    planner_model.eval()

    past = token_blob["past"].float()
    future = token_blob["future"].float()

    z_all = encode_tensor_batchwise(
        sae,
        token_tensor,
        batch_size=args.batch_size,
        device=device,
    )
    latent_dim = sae_ckpt["latent_dim"]
    features = clip_feature_list(
        parse_feature_spec(args.features, latent_dim=latent_dim),
        latent_dim=latent_dim,
        feature_start=args.feature_start,
        feature_end=args.feature_end,
    )

    active_mask = z_all > 0
    active_count = active_mask.sum(dim=0)
    active_sum = z_all.sum(dim=0)
    active_sum_sq = (z_all * z_all).sum(dim=0)
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

    rows = []
    for random_seed in parse_int_list(args.random_seeds):
        for feature_idx in features:
            row = summarize_feature(
                feature_idx=feature_idx,
                token_tensor_cpu=token_tensor,
                z_all=z_all,
                scales=scales,
                sae=sae,
                planner_model=planner_model,
                lit_model=lit_model,
                sae_block=args.sae_block,
                past=past,
                future=future,
                relevant_scenes=args.relevant_scenes_per_feature,
                alphas=parse_float_list(args.alphas),
                scene_selection=args.scene_selection,
                random_seed=random_seed,
                dataset=dataset,
                device=device,
            )
            rows.append(row)
            print(row)

    rows.sort(key=lambda row: row["mean_delta_selected_ade"])
    write_csv(Path(args.output_csv), rows)
    print(f"Used token key {token_key}")
    print(f"Saved CSV to {args.output_csv}")
