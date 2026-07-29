import argparse
import csv
from pathlib import Path

import torch # type: ignore

from sae_utils import (
    build_sae_from_checkpoint,
    default_analysis_dir,
    default_device,
    encode_tensor_batchwise,
    load_sae_bundle,
    resolve_token_tensor,
)


INTENT_NAMES = {
    0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
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


def compute_trajectory_stats(token_blob: dict) -> dict[str, torch.Tensor]:
    stats = {}

    if "future" in token_blob:
        future = token_blob["future"].float()
        displacement = future[:, -1] - future[:, 0]
        stats["future_final_x"] = future[:, -1, 0]
        stats["future_final_y"] = future[:, -1, 1]
        stats["future_total_displacement"] = torch.norm(displacement, dim=-1)

        if future.shape[1] >= 3:
            seg = future[:, 1:] - future[:, :-1]
            seg_norm = torch.norm(seg, dim=-1).clamp_min(1e-6)
            heading = torch.atan2(seg[..., 1], seg[..., 0])
            d_heading = torch.atan2(
                torch.sin(heading[:, 1:] - heading[:, :-1]),
                torch.cos(heading[:, 1:] - heading[:, :-1]),
            ).abs()
            ds = 0.5 * (seg_norm[:, 1:] + seg_norm[:, :-1])
            curvature = d_heading / ds.clamp_min(1e-6)
            stats["future_avg_curvature"] = curvature.mean(dim=1)

    if "scores" in token_blob:
        scores = token_blob["scores"].float()
        selected_idx = scores.argmin(dim=1)
        stats["selected_proposal"] = selected_idx.float()

        if scores.shape[1] > 1:
            sorted_scores = scores.sort(dim=1).values
            stats["score_margin"] = sorted_scores[:, 1] - sorted_scores[:, 0]

        probs = torch.softmax(-scores, dim=1)
        stats["score_entropy"] = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)

    if "trajectory" in token_blob and "scores" in token_blob and "future" in token_blob:
        future = token_blob["future"].float()
        trajectory = token_blob["trajectory"].float()
        scores = token_blob["scores"].float()

        n = future.shape[0]
        horizon = future.shape[1]
        num_proposals = scores.shape[1]

        traj = trajectory.view(n, num_proposals, horizon, 2)
        dist = torch.norm(traj - future[:, None], dim=-1)
        ade_per_mode = dist.mean(dim=-1)

        selected_idx = scores.argmin(dim=1)
        row_idx = torch.arange(n)

        selected_ade = ade_per_mode[row_idx, selected_idx]
        oracle_ade = ade_per_mode.min(dim=1).values
        regret = selected_ade - oracle_ade

        traj_mean = traj.mean(dim=1, keepdim=True)
        proposal_spread = torch.norm(traj - traj_mean, dim=-1).mean(dim=(1, 2))

        stats["selected_ade"] = selected_ade
        stats["oracle_ade"] = oracle_ade
        stats["regret"] = regret
        stats["proposal_spread"] = proposal_spread

    if "controls" in token_blob and "scores" in token_blob and "future" in token_blob:
        controls = token_blob["controls"].float()
        scores = token_blob["scores"].float()
        future = token_blob["future"].float()

        n = future.shape[0]
        horizon = future.shape[1]
        num_proposals = scores.shape[1]

        ctrl = controls.view(n, num_proposals, horizon, 2)
        selected_idx = scores.argmin(dim=1)
        row_idx = torch.arange(n)
        selected_ctrl = ctrl[row_idx, selected_idx]

        accel = selected_ctrl[..., 0]
        stats["brake_mag"] = (-accel).clamp_min(0).mean(dim=1)
        stats["accel_mag"] = accel.clamp_min(0).mean(dim=1)

    if "past" in token_blob:
        past = token_blob["past"].float()  # (N, 16, 6)
    
        vx = past[:, -1, 2]
        vy = past[:, -1, 3]
        ax = past[:, -1, 4]
        ay = past[:, -1, 5]
        
        speed = torch.sqrt(vx * vx + vy * vy)

        stats["past_final_speed"] = torch.sqrt(vx * vx + vy * vy)
        stats["past_final_heading"] = torch.atan2(vy, vx)
        stats["past_final_accel_mag"] = torch.sqrt(ax * ax + ay * ay)
    
        # curvature of past trajectory
        seg = past[:, 1:, :2] - past[:, :-1, :2]  # (N, 15, 2)
        seg_norm = torch.norm(seg, dim=-1).clamp_min(1e-6)  # (N, 15)
        heading = torch.atan2(seg[..., 1], seg[..., 0])  # (N, 15)
        d_heading = torch.atan2(
            torch.sin(heading[:, 1:] - heading[:, :-1]),
            torch.cos(heading[:, 1:] - heading[:, :-1]),
        ).abs()  # (N, 14)
        ds = 0.5 * (seg_norm[:, 1:] + seg_norm[:, :-1])  # (N, 14)
        curvature = d_heading / ds.clamp_min(1e-6)
        mean_curvature = curvature.mean(dim=1)

        # mask out near-stationary scenes
        moving_mask = speed > 0.5
        stats["past_avg_curvature"] = torch.where(moving_mask, mean_curvature, torch.zeros_like(mean_curvature))

        # stats["past_avg_curvature"] = (d_heading / ds.clamp_min(1e-6)).mean(dim=1)
    
        # was the vehicle braking or accelerating over the past window?
        accel_x = past[:, :, 4]  # (N, 16)
        stats["past_mean_accel"] = accel_x.mean(dim=1)
        stats["past_mean_brake"] = (-accel_x).clamp_min(0).mean(dim=1)

    return stats


def compute_feature_correlation(
    z_all: torch.Tensor,
    min_active_frac: float,
    max_active_frac: float,
    use_binary: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = z_all > 0
    active_frac = active.float().mean(dim=0)

    keep_mask = (active_frac >= min_active_frac) & (active_frac <= max_active_frac)
    kept_features = torch.where(keep_mask)[0]

    if kept_features.numel() == 0:
        raise ValueError(
            "No features survived active-frequency filtering. "
            "Try lowering --min_active_frac or raising --max_active_frac."
        )

    if use_binary:
        x = active[:, keep_mask].float()
    else:
        x = z_all[:, keep_mask].float()

    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True, unbiased=False) + 1e-6)

    corr = (x.T @ x) / x.shape[0]
    corr.fill_diagonal_(0.0)

    return corr.cpu(), kept_features.cpu(), active_frac.cpu()


def connected_components_from_corr(
    corr: torch.Tensor,
    kept_features: torch.Tensor,
    threshold: float,
    min_cluster_size: int,
) -> list[list[int]]:
    n = corr.shape[0]
    adjacency = corr >= threshold

    visited = torch.zeros(n, dtype=torch.bool)
    clusters = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while stack:
            node = stack.pop()
            component.append(int(kept_features[node].item()))

            neighbors = torch.where(adjacency[node] & ~visited)[0].tolist()
            for nb in neighbors:
                visited[nb] = True
                stack.append(nb)

        if len(component) >= min_cluster_size:
            clusters.append(sorted(component))

    clusters.sort(key=len, reverse=True)
    return clusters


def mean_pairwise_corr_for_cluster(
    features: list[int],
    feature_to_corr_idx: dict[int, int],
    corr: torch.Tensor,
) -> float:
    corr_indices = [feature_to_corr_idx[f] for f in features if f in feature_to_corr_idx]
    if len(corr_indices) < 2:
        return 0.0

    idx = torch.tensor(corr_indices, dtype=torch.long)
    sub = corr[idx][:, idx]
    i, j = torch.triu_indices(len(idx), len(idx), offset=1)
    return float(sub[i, j].mean().item())


def summarize_cluster(
    cluster_id: int,
    features: list[int],
    z_all: torch.Tensor,
    active_frac: torch.Tensor,
    corr: torch.Tensor,
    feature_to_corr_idx: dict[int, int],
    top_k_scenes: int,
    token_blob: dict,
    trajectory_stats: dict[str, torch.Tensor],
) -> tuple[dict, list[dict]]:
    feature_tensor = torch.tensor(features, dtype=torch.long)

    cluster_z = z_all[:, feature_tensor]
    cluster_score = cluster_z.sum(dim=1)
    cluster_active_count = (cluster_z > 0).sum(dim=1)
    cluster_any_active = cluster_active_count > 0

    k = min(top_k_scenes, z_all.shape[0])
    top_vals, top_idx = torch.topk(cluster_score, k=k)

    summary = {
        "cluster_id": cluster_id,
        "num_features": len(features),
        "features": " ".join(map(str, features)),
        "mean_pairwise_corr": mean_pairwise_corr_for_cluster(
            features=features,
            feature_to_corr_idx=feature_to_corr_idx,
            corr=corr,
        ),
        "mean_feature_active_frac": float(active_frac[feature_tensor].mean().item()),
        "min_feature_active_frac": float(active_frac[feature_tensor].min().item()),
        "max_feature_active_frac": float(active_frac[feature_tensor].max().item()),
        "scene_any_feature_active_frac": float(cluster_any_active.float().mean().item()),
        "mean_cluster_score": float(cluster_score.mean().item()),
        "max_cluster_score": float(cluster_score.max().item()),
    }

    if "intent" in token_blob:
        intent = token_blob["intent"].long()
        top_intent = intent[top_idx]
        for intent_id in sorted(torch.unique(intent).tolist()):
            name = INTENT_NAMES.get(int(intent_id), str(int(intent_id)))
            summary[f"top_frac_intent_{name}"] = float(
                (top_intent == intent_id).float().mean().item()
            )

    for stat_name, values in trajectory_stats.items():
        summary[f"top_mean_{stat_name}"] = float(values[top_idx].float().mean().item())
        summary[f"global_mean_{stat_name}"] = float(values.float().mean().item())
        summary[f"global_std_{stat_name}"] = float(values.float().std().item())
        summary[f"zscore_{stat_name}"] = (
            (values[top_idx].float().mean() - values.float().mean()) / (values.float().std() + 1e-6)
        ).item()

    top_rows = []

    for rank, scene_idx in enumerate(top_idx.tolist(), start=1):
        active_features = [
            str(f)
            for f in features
            if float(z_all[scene_idx, f].item()) > 0
        ]

        row = {
            "cluster_id": cluster_id,
            "rank": rank,
            "scene_idx": scene_idx,
            "cluster_score": float(cluster_score[scene_idx].item()),
            "num_active_cluster_features": int(cluster_active_count[scene_idx].item()),
            "active_cluster_features": " ".join(active_features),
        }

        if "intent" in token_blob:
            intent_id = int(token_blob["intent"][scene_idx].item())
            row["intent_id"] = intent_id
            row["intent_name"] = INTENT_NAMES.get(intent_id, str(intent_id))

        for stat_name, values in trajectory_stats.items():
            row[stat_name] = float(values[scene_idx].item())

        top_rows.append(row)

    return summary, top_rows


def write_feature_pair_csv(
    path: Path,
    corr: torch.Tensor,
    kept_features: torch.Tensor,
    active_frac: torch.Tensor,
    top_k_pairs: int,
) -> None:
    rows = []

    n = corr.shape[0]
    if n < 2:
        write_csv(path, rows)
        return

    triu_i, triu_j = torch.triu_indices(n, n, offset=1)
    vals = corr[triu_i, triu_j]

    k = min(top_k_pairs, vals.numel())
    top_vals, top_pos = torch.topk(vals, k=k)

    for rank, pos in enumerate(top_pos.tolist(), start=1):
        i = int(triu_i[pos].item())
        j = int(triu_j[pos].item())

        feature_i = int(kept_features[i].item())
        feature_j = int(kept_features[j].item())

        rows.append(
            {
                "rank": rank,
                "feature_i": feature_i,
                "feature_j": feature_j,
                "corr": float(top_vals[rank - 1].item()),
                "feature_i_active_frac": float(active_frac[feature_i].item()),
                "feature_j_active_frac": float(active_frac[feature_j].item()),
            }
        )

    write_csv(path, rows)


def write_feature_frequency_csv(
    path: Path,
    active_frac: torch.Tensor,
) -> None:
    rows = []
    for feature_idx in range(active_frac.numel()):
        rows.append(
            {
                "feature_idx": feature_idx,
                "active_frac": float(active_frac[feature_idx].item()),
            }
        )

    rows.sort(key=lambda row: row["active_frac"], reverse=True)
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find groups of SAE features that co-activate and summarize the scenes that trigger them."
    )

    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--sae_block", type=int, default=3)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument(
        "--use_binary",
        action="store_true",
        help="Use binary firing indicators instead of raw SAE activation magnitudes.",
    )
    parser.add_argument("--min_active_frac", type=float, default=0.005)
    parser.add_argument("--max_active_frac", type=float, default=0.50)
    parser.add_argument("--corr_threshold", type=float, default=0.4)
    parser.add_argument("--min_cluster_size", type=int, default=3)

    parser.add_argument("--top_k_scenes", type=int, default=50)
    parser.add_argument("--top_k_pairs", type=int, default=500)

    args = parser.parse_args()

    device = torch.device(args.device)
    run_root = Path(args.run_root)

    output_dir = default_analysis_dir(run_root, args.sae_block, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAE bundle...")
    bundle = load_sae_bundle(
        run_root,
        args.split,
        args.sae_block,
        map_location="cpu",
    )

    sae_ckpt = bundle["ckpt"]
    token_blob = bundle["token_blob"]

    print(f"SAE checkpoint path: {bundle['ckpt_path']}")
    print(f"Token path: {bundle['token_path']}")

    print("Resolving token tensor...")
    token_tensor, token_key = resolve_token_tensor(token_blob, args.sae_block)
    print(f"Using token key: {token_key}")
    print(f"Token tensor shape: {tuple(token_tensor.shape)}")

    print("Building SAE...")
    sae = build_sae_from_checkpoint(sae_ckpt, bundle["legacy_norm"])
    sae.to(device)
    sae.eval()

    print("Encoding SAE activations...")
    z_all = encode_tensor_batchwise(
        sae,
        token_tensor,
        batch_size=args.batch_size,
        device=device,
    ).cpu()

    print(f"SAE activation tensor z_all shape: {tuple(z_all.shape)}")

    print("Computing auxiliary scene/output statistics...")
    trajectory_stats = compute_trajectory_stats(token_blob)

    print("Computing feature correlations...")
    corr, kept_features, active_frac = compute_feature_correlation(
        z_all=z_all,
        min_active_frac=args.min_active_frac,
        max_active_frac=args.max_active_frac,
        use_binary=args.use_binary,
    )

    print(f"Kept {len(kept_features)} / {z_all.shape[1]} features after active-frequency filtering.")

    feature_to_corr_idx = {
        int(feature.item()): idx
        for idx, feature in enumerate(kept_features)
    }

    print("Finding correlated feature clusters...")
    clusters = connected_components_from_corr(
        corr=corr,
        kept_features=kept_features,
        threshold=args.corr_threshold,
        min_cluster_size=args.min_cluster_size,
    )

    print(f"Found {len(clusters)} clusters.")

    freq_csv = output_dir / f"sae_feature_frequencies_block_{args.sae_block}_{args.split}_thresh{args.corr_threshold}.csv"
    pair_csv = output_dir / f"sae_top_correlated_pairs_block_{args.sae_block}_{args.split}_thresh{args.corr_threshold}.csv"
    cluster_csv = output_dir / f"sae_correlation_clusters_block_{args.sae_block}_{args.split}_thresh{args.corr_threshold}.csv"
    top_scene_csv = output_dir / f"sae_cluster_top_scenes_block_{args.sae_block}_{args.split}_thresh{args.corr_threshold}.csv"
    print("Writing feature frequencies...")
    write_feature_frequency_csv(freq_csv, active_frac)

    print("Writing top correlated feature pairs...")
    write_feature_pair_csv(
        path=pair_csv,
        corr=corr,
        kept_features=kept_features,
        active_frac=active_frac,
        top_k_pairs=args.top_k_pairs,
    )

    print("Summarizing clusters...")
    cluster_rows = []
    top_scene_rows = []

    for cluster_id, features in enumerate(clusters):
        summary, scene_rows = summarize_cluster(
            cluster_id=cluster_id,
            features=features,
            z_all=z_all,
            active_frac=active_frac,
            corr=corr,
            feature_to_corr_idx=feature_to_corr_idx,
            top_k_scenes=args.top_k_scenes,
            token_blob=token_blob,
            trajectory_stats=trajectory_stats,
        )

        cluster_rows.append(summary)
        top_scene_rows.extend(scene_rows)

    write_csv(cluster_csv, cluster_rows)
    write_csv(top_scene_csv, top_scene_rows)

    print("")
    print("Done.")
    print(f"Saved feature frequencies to: {freq_csv}")
    print(f"Saved top correlated pairs to: {pair_csv}")
    print(f"Saved cluster summaries to: {cluster_csv}")
    print(f"Saved top scenes per cluster to: {top_scene_csv}")

    print("")
    print("Interpretation step:")
    print("1. Open the cluster summaries CSV.")
    print("2. Pick clusters with high mean_pairwise_corr and many features.")
    print("3. Open the top-scenes CSV for those cluster IDs.")
    print("4. Visualize those scene_idx values using the repo's existing visualization tools.")
    print("5. Write a hypothesis for what input/scenario property causes the cluster to activate.")


if __name__ == "__main__":
    main()
