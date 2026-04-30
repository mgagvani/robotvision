import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from extract_planner_tok import load_model
from loader import WaymoE2E
from models.base_model import collate_with_images
from sae_utils import (
    build_sae_from_checkpoint,
    default_analysis_dir,
    default_device,
    encode_tensor_batchwise,
    load_sae_bundle,
    planner_token_key,
    resolve_token_tensor,
)


PLANNER_DELTA_NAMES = (
    "trajectory_l2_delta",
    "delta_selected_ade",
    "delta_oracle_ade",
    "delta_score_margin",
    "delta_proposal_spread",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_checkpoint_path(run_root: Path, token_blob: dict, explicit_path: str | None) -> str:
    if explicit_path is not None:
        return explicit_path
    meta_path = token_blob.get("meta", {}).get("checkpoint")
    if meta_path and Path(meta_path).exists():
        return meta_path
    repo_default = Path(__file__).resolve().parent / "camera-e2e-epoch=04-val_loss=2.90.ckpt"
    if repo_default.exists():
        return str(repo_default)
    raise FileNotFoundError("Could not infer planner checkpoint path. Pass --planner_checkpoint.")


def resolve_edited_path(manifest_path: Path, row: dict) -> Path:
    edited_value = row.get("edited_path")
    if not edited_value:
        raise ValueError(f"Missing edited_path for dataset_idx={row.get('dataset_idx')}")
    edited_path = Path(edited_value)
    if not edited_path.is_absolute():
        edited_path = manifest_path.parent / edited_path
    return edited_path


def jpeg_bytes_to_tensor(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    return torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())


def load_manifest_rows(
    manifest_path: Path,
    *,
    strict_manifest: bool,
) -> list[dict]:
    rows = []
    for row in read_jsonl(manifest_path):
        if row.get("status") != "edited":
            continue
        edited_path = resolve_edited_path(manifest_path, row)
        if not edited_path.exists():
            message = f"Missing edited image for dataset_idx={row.get('dataset_idx')}: {edited_path}"
            if strict_manifest:
                raise FileNotFoundError(message)
            print(f"WARNING: {message}; skipping")
            continue
        row = dict(row)
        row["resolved_edited_path"] = str(edited_path)
        rows.append(row)
    return rows


def choose_manifest_value(rows: list[dict], key: str, override):
    if override is not None:
        return override
    values = {row.get(key) for row in rows if row.get(key) is not None}
    if len(values) == 1:
        return values.pop()
    if not values:
        return None
    raise ValueError(f"Manifest has multiple {key} values; pass --{key} explicitly.")


def build_token_name_lookup(token_blob: dict) -> dict[str, int]:
    names = token_blob.get("names")
    if names is None:
        raise KeyError("Token blob is missing names; cannot align original tokens to manifest rows.")

    lookup = {}
    duplicates = set()
    for idx, name in enumerate(names):
        if name in lookup:
            duplicates.add(name)
        lookup[name] = idx
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Token blob has duplicate sample names, cannot build unambiguous lookup: {examples}")
    return lookup


def make_pair_samples(
    dataset: WaymoE2E,
    rows: list[dict],
    *,
    allow_name_mismatch: bool,
) -> tuple[list[dict], list[dict]]:
    orig_samples = []
    edited_samples = []
    for row in rows:
        dataset_idx = int(row["dataset_idx"])
        sample = dataset[dataset_idx]
        if not allow_name_mismatch and row.get("name") != sample["NAME"]:
            raise ValueError(
                f"Manifest name mismatch at dataset_idx={dataset_idx}: "
                f"manifest={row.get('name')} dataset={sample['NAME']}"
            )

        camera_idx = int(row.get("camera_idx", 1))
        edited_sample = {
            "PAST": sample["PAST"],
            "FUTURE": sample["FUTURE"],
            "INTENT": sample["INTENT"],
            "NAME": sample["NAME"],
            "IMAGES_JPEG": list(sample["IMAGES_JPEG"]),
        }
        edited_sample["IMAGES_JPEG"][camera_idx] = jpeg_bytes_to_tensor(Path(row["resolved_edited_path"]))
        orig_samples.append(sample)
        edited_samples.append(edited_sample)
    return orig_samples, edited_samples


def selected_and_oracle_ade(
    trajectory: torch.Tensor,
    scores: torch.Tensor,
    future: torch.Tensor,
    *,
    num_proposals: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = future.size(0)
    traj = trajectory.view(batch, num_proposals, horizon, 2)
    dist = torch.norm(traj - future[:, None], dim=-1)
    ade_per_mode = dist.mean(dim=-1)
    selected_idx = scores.argmin(dim=1)
    row_idx = torch.arange(batch, device=future.device)
    selected_ade = ade_per_mode[row_idx, selected_idx]
    oracle_ade = ade_per_mode.min(dim=1).values
    return selected_idx, selected_ade, oracle_ade


def score_margin(scores: torch.Tensor) -> torch.Tensor:
    sorted_scores = scores.sort(dim=1).values
    if scores.size(1) < 2:
        return torch.zeros(scores.size(0), device=scores.device, dtype=scores.dtype)
    return sorted_scores[:, 1] - sorted_scores[:, 0]


def proposal_spread(trajectory: torch.Tensor, *, num_proposals: int, horizon: int) -> torch.Tensor:
    batch = trajectory.size(0)
    traj = trajectory.view(batch, num_proposals, horizon, 2)
    mean_traj = traj.mean(dim=1, keepdim=True)
    return torch.norm(traj - mean_traj, dim=-1).mean(dim=(1, 2))


def controls_view(controls: torch.Tensor, *, num_proposals: int, horizon: int) -> torch.Tensor:
    return controls.view(controls.size(0), num_proposals, horizon, 2)


def compute_pair_metrics(
    out_orig: dict,
    out_edit: dict,
    future: torch.Tensor,
    *,
    num_proposals: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    selected_idx_orig, selected_ade_orig, oracle_ade_orig = selected_and_oracle_ade(
        out_orig["trajectory"],
        out_orig["scores"],
        future,
        num_proposals=num_proposals,
        horizon=horizon,
    )
    selected_idx_edit, selected_ade_edit, oracle_ade_edit = selected_and_oracle_ade(
        out_edit["trajectory"],
        out_edit["scores"],
        future,
        num_proposals=num_proposals,
        horizon=horizon,
    )

    score_margin_orig = score_margin(out_orig["scores"])
    score_margin_edit = score_margin(out_edit["scores"])
    spread_orig = proposal_spread(out_orig["trajectory"], num_proposals=num_proposals, horizon=horizon)
    spread_edit = proposal_spread(out_edit["trajectory"], num_proposals=num_proposals, horizon=horizon)

    ctrl_orig = controls_view(out_orig["controls"], num_proposals=num_proposals, horizon=horizon)
    ctrl_edit = controls_view(out_edit["controls"], num_proposals=num_proposals, horizon=horizon)
    ctrl_delta = (ctrl_edit - ctrl_orig).abs()

    return {
        "trajectory_l2_delta": torch.norm(out_edit["trajectory"] - out_orig["trajectory"], dim=1),
        "selected_idx_orig": selected_idx_orig,
        "selected_idx_edit": selected_idx_edit,
        "selected_ade_orig": selected_ade_orig,
        "selected_ade_edit": selected_ade_edit,
        "delta_selected_ade": selected_ade_edit - selected_ade_orig,
        "oracle_ade_orig": oracle_ade_orig,
        "oracle_ade_edit": oracle_ade_edit,
        "delta_oracle_ade": oracle_ade_edit - oracle_ade_orig,
        "score_margin_orig": score_margin_orig,
        "score_margin_edit": score_margin_edit,
        "delta_score_margin": score_margin_edit - score_margin_orig,
        "proposal_spread_orig": spread_orig,
        "proposal_spread_edit": spread_edit,
        "delta_proposal_spread": spread_edit - spread_orig,
        "mean_abs_accel_delta": ctrl_delta[..., 0].mean(dim=(1, 2)),
        "mean_abs_omega_delta": ctrl_delta[..., 1].mean(dim=(1, 2)),
    }


def model_inputs_from_batch(batch: dict, *, lit_model, device: torch.device) -> dict:
    return {
        "PAST": batch["PAST"].to(device, non_blocking=True),
        "IMAGES": lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device),
        "INTENT": batch["INTENT"].to(device, non_blocking=True),
    }


def compute_baseline_stats(
    *,
    sae,
    token_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
    min_scale: float,
) -> dict[str, torch.Tensor]:
    z_all = encode_tensor_batchwise(sae, token_tensor, batch_size=batch_size, device=device)
    all_mean = z_all.mean(dim=0)
    all_std = z_all.std(dim=0, unbiased=False)
    active_mask = z_all > 0
    active_count = active_mask.sum(dim=0)
    active_sum = z_all.sum(dim=0)
    active_sum_sq = (z_all * z_all).sum(dim=0)
    denom = active_count.clamp_min(1)
    active_mean = active_sum / denom
    active_var = active_sum_sq / denom - active_mean.square()
    active_std = torch.sqrt(active_var.clamp_min(0.0))
    feature_scale = torch.maximum(active_std, 0.25 * active_mean).clamp_min(min_scale)
    return {
        "all_mean": all_mean,
        "all_std": all_std.clamp_min(1e-6),
        "active_count": active_count,
        "active_rate": active_count.float() / max(1, z_all.size(0)),
        "active_mean": active_mean,
        "active_std": active_std,
        "feature_scale": feature_scale,
    }


def replay_pairs(
    *,
    rows: list[dict],
    dataset: WaymoE2E,
    planner_model,
    lit_model,
    sae,
    sae_block: int,
    token_blob: dict,
    token_tensor: torch.Tensor,
    token_name_to_idx: dict[str, int],
    batch_size: int,
    device: torch.device,
    allow_name_mismatch: bool,
) -> dict:
    token_key = planner_token_key(sae_block)
    pair_rows = []
    tokens_orig_chunks = []
    tokens_edit_chunks = []
    z_orig_chunks = []
    z_edit_chunks = []
    planner_metric_chunks = {name: [] for name in (
        "trajectory_l2_delta",
        "selected_idx_orig",
        "selected_idx_edit",
        "selected_ade_orig",
        "selected_ade_edit",
        "delta_selected_ade",
        "oracle_ade_orig",
        "oracle_ade_edit",
        "delta_oracle_ade",
        "score_margin_orig",
        "score_margin_edit",
        "delta_score_margin",
        "proposal_spread_orig",
        "proposal_spread_edit",
        "delta_proposal_spread",
        "mean_abs_accel_delta",
        "mean_abs_omega_delta",
    )}

    planner_model.eval()
    sae.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            _, edited_samples = make_pair_samples(
                dataset,
                batch_rows,
                allow_name_mismatch=allow_name_mismatch,
            )
            edit_batch = collate_with_images(edited_samples)
            token_indices = []
            for row in batch_rows:
                name = row.get("name")
                if name not in token_name_to_idx:
                    raise KeyError(f"Manifest sample name is not present in token blob: {name}")
                token_indices.append(token_name_to_idx[name])
            token_indices = torch.tensor(token_indices, dtype=torch.long)

            token_orig_cpu = token_tensor.index_select(0, token_indices)
            token_orig = token_orig_cpu.to(device, non_blocking=True)
            out_orig = {
                "trajectory": token_blob["trajectory"].index_select(0, token_indices).to(device, non_blocking=True),
                "scores": token_blob["scores"].index_select(0, token_indices).to(device, non_blocking=True),
                "controls": token_blob["controls"].index_select(0, token_indices).to(device, non_blocking=True),
            }
            future = token_blob["future"].index_select(0, token_indices).to(device, non_blocking=True)
            out_edit = planner_model(
                model_inputs_from_batch(edit_batch, lit_model=lit_model, device=device),
                return_block_tokens=True,
            )

            token_edit = out_edit[token_key]
            z_orig = sae.encode(token_orig)
            z_edit = sae.encode(token_edit)
            metrics = compute_pair_metrics(
                out_orig,
                out_edit,
                future,
                num_proposals=planner_model.n_proposals,
                horizon=planner_model.horizon,
            )

            tokens_orig_chunks.append(token_orig.detach().cpu())
            tokens_edit_chunks.append(token_edit.detach().cpu())
            z_orig_chunks.append(z_orig.detach().cpu())
            z_edit_chunks.append(z_edit.detach().cpu())
            for name, value in metrics.items():
                planner_metric_chunks[name].append(value.detach().cpu())

            token_delta_norm = torch.norm(token_edit - token_orig, dim=1).detach().cpu()
            latent_delta_norm = torch.norm(z_edit - z_orig, dim=1).detach().cpu()
            for i, row in enumerate(batch_rows):
                pair_row = {
                    "dataset_idx": int(row["dataset_idx"]),
                    "name": row.get("name"),
                    "camera_idx": int(row.get("camera_idx", 1)),
                    "edit_type": row.get("edit_type"),
                    "edit_direction": row.get("edit_direction"),
                    "edited_path": row["resolved_edited_path"],
                    "token_blob_idx": int(token_indices[i].item()),
                    "original_source": "token_blob",
                    "token_l2_delta": float(token_delta_norm[i].item()),
                    "latent_l2_delta": float(latent_delta_norm[i].item()),
                    "latent_mean_abs_delta": float((z_edit[i] - z_orig[i]).abs().mean().item()),
                }
                for metric_name, metric_value in metrics.items():
                    value = metric_value[i]
                    if value.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.long}:
                        pair_row[metric_name] = int(value.item())
                    else:
                        pair_row[metric_name] = float(value.item())
                pair_rows.append(pair_row)

            print(f"Processed {min(start + batch_size, len(rows))}/{len(rows)} edited pairs", flush=True)

    return {
        "pair_rows": pair_rows,
        "tokens_orig": torch.cat(tokens_orig_chunks, dim=0),
        "tokens_edit": torch.cat(tokens_edit_chunks, dim=0),
        "z_orig": torch.cat(z_orig_chunks, dim=0),
        "z_edit": torch.cat(z_edit_chunks, dim=0),
        "planner_metrics": {
            name: torch.cat(chunks, dim=0) for name, chunks in planner_metric_chunks.items()
        },
    }


def pearson_vector(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2:
        return torch.zeros(x.size(1), dtype=torch.float32)
    x = x.float()
    y = y.float()
    x_center = x - x.mean(dim=0, keepdim=True)
    y_center = y - y.mean()
    x_denom = torch.sqrt((x_center * x_center).sum(dim=0)).clamp_min(1e-6)
    y_denom = torch.sqrt((y_center * y_center).sum()).clamp_min(1e-6)
    return (x_center * y_center[:, None]).sum(dim=0) / (x_denom * y_denom)


def summarize_group(
    *,
    group_kind: str,
    group_value: str,
    mask: torch.Tensor,
    z_orig: torch.Tensor,
    z_edit: torch.Tensor,
    baseline_stats: dict[str, torch.Tensor],
    planner_metrics: dict[str, torch.Tensor],
) -> list[dict]:
    z_o = z_orig[mask]
    z_e = z_edit[mask]
    delta = z_e - z_o
    abs_delta = delta.abs()
    n = delta.size(0)
    latent_dim = delta.size(1)
    if n == 0:
        return []

    mean_delta = delta.mean(dim=0)
    mean_abs_delta = abs_delta.mean(dim=0)
    std_delta = delta.std(dim=0, unbiased=False)
    stderr_delta = std_delta / (n ** 0.5)
    paired_t = mean_delta / stderr_delta.clamp_min(1e-6)
    cohen_dz = mean_delta / std_delta.clamp_min(1e-6)
    sign_consistency = torch.where(
        mean_delta >= 0,
        (delta > 0).float().mean(dim=0),
        (delta < 0).float().mean(dim=0),
    )

    orig_active = z_o > 0
    edit_active = z_e > 0
    orig_active_rate = orig_active.float().mean(dim=0)
    edited_active_rate = edit_active.float().mean(dim=0)
    flip_on_rate = ((~orig_active) & edit_active).float().mean(dim=0)
    flip_off_rate = (orig_active & (~edit_active)).float().mean(dim=0)

    feature_scale = baseline_stats["feature_scale"]
    all_std = baseline_stats["all_std"]
    mean_delta_scale_units = mean_delta / feature_scale
    mean_abs_delta_scale_units = mean_abs_delta / feature_scale
    mean_delta_all_std_units = mean_delta / all_std
    mean_abs_delta_all_std_units = mean_abs_delta / all_std

    corr_delta_selected = pearson_vector(delta, planner_metrics["delta_selected_ade"][mask])
    corr_abs_traj = pearson_vector(abs_delta, planner_metrics["trajectory_l2_delta"][mask])
    corr_delta_score_margin = pearson_vector(delta, planner_metrics["delta_score_margin"][mask])

    rows = []
    for feature_idx in range(latent_dim):
        rows.append(
            {
                "group_kind": group_kind,
                "group_value": group_value,
                "feature_idx": feature_idx,
                "n": n,
                "mean_delta": float(mean_delta[feature_idx].item()),
                "mean_abs_delta": float(mean_abs_delta[feature_idx].item()),
                "std_delta": float(std_delta[feature_idx].item()),
                "stderr_delta": float(stderr_delta[feature_idx].item()),
                "paired_t": float(paired_t[feature_idx].item()),
                "cohen_dz": float(cohen_dz[feature_idx].item()),
                "mean_delta_scale_units": float(mean_delta_scale_units[feature_idx].item()),
                "mean_abs_delta_scale_units": float(mean_abs_delta_scale_units[feature_idx].item()),
                "mean_delta_all_std_units": float(mean_delta_all_std_units[feature_idx].item()),
                "mean_abs_delta_all_std_units": float(mean_abs_delta_all_std_units[feature_idx].item()),
                "baseline_active_rate": float(baseline_stats["active_rate"][feature_idx].item()),
                "orig_active_rate": float(orig_active_rate[feature_idx].item()),
                "edited_active_rate": float(edited_active_rate[feature_idx].item()),
                "flip_on_rate": float(flip_on_rate[feature_idx].item()),
                "flip_off_rate": float(flip_off_rate[feature_idx].item()),
                "sign_consistency": float(sign_consistency[feature_idx].item()),
                "corr_delta_z_delta_selected_ade": float(corr_delta_selected[feature_idx].item()),
                "corr_abs_delta_z_trajectory_l2_delta": float(corr_abs_traj[feature_idx].item()),
                "corr_delta_z_delta_score_margin": float(corr_delta_score_margin[feature_idx].item()),
            }
        )
    return rows


def build_summary_rows(
    *,
    rows: list[dict],
    z_orig: torch.Tensor,
    z_edit: torch.Tensor,
    baseline_stats: dict[str, torch.Tensor],
    planner_metrics: dict[str, torch.Tensor],
) -> list[dict]:
    out = []
    for group_kind in ("edit_type", "edit_direction"):
        values = sorted({str(row.get(group_kind)) for row in rows})
        for value in values:
            mask = torch.tensor([str(row.get(group_kind)) == value for row in rows], dtype=torch.bool)
            out.extend(
                summarize_group(
                    group_kind=group_kind,
                    group_value=value,
                    mask=mask,
                    z_orig=z_orig,
                    z_edit=z_edit,
                    baseline_stats=baseline_stats,
                    planner_metrics=planner_metrics,
                )
            )
    return out


def save_feature_deltas_pt(
    path: Path,
    *,
    rows: list[dict],
    z_orig: torch.Tensor,
    z_edit: torch.Tensor,
    baseline_stats: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    delta = z_edit - z_orig
    torch.save(
        {
            "rows": rows,
            "z_orig": z_orig,
            "z_edit": z_edit,
            "delta_z": delta,
            "abs_delta_z": delta.abs(),
            "delta_z_scale_units": delta / baseline_stats["feature_scale"],
            "delta_z_all_std_units": delta / baseline_stats["all_std"],
            "orig_active": z_orig > 0,
            "edited_active": z_edit > 0,
            "flip_on": (z_orig <= 0) & (z_edit > 0),
            "flip_off": (z_orig > 0) & (z_edit <= 0),
            "feature_scale": baseline_stats["feature_scale"],
            "all_std": baseline_stats["all_std"],
            "baseline_active_rate": baseline_stats["active_rate"],
        },
        path,
    )


def build_top_feature_rows(summary_rows: list[dict], top_k: int) -> list[dict]:
    grouped = {}
    for row in summary_rows:
        grouped.setdefault((row["group_kind"], row["group_value"]), []).append(row)
    top_rows = []
    for (group_kind, group_value), group_rows in sorted(grouped.items()):
        ranked = sorted(
            group_rows,
            key=lambda row: abs(row["mean_delta_scale_units"]),
            reverse=True,
        )[:top_k]
        for rank, row in enumerate(ranked, start=1):
            top_row = dict(row)
            top_row["rank"] = rank
            top_rows.append(top_row)
    return top_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--planner_checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--sae_block", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--encode_batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--n_items", type=int, default=None)
    parser.add_argument("--min_scale", type=float, default=0.05)
    parser.add_argument("--allow_name_mismatch", action="store_true")
    parser.add_argument("--strict_manifest", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    run_root = Path(args.run_root)
    output_dir = default_analysis_dir(run_root, args.sae_block, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows = load_manifest_rows(manifest_path, strict_manifest=args.strict_manifest)
    if not rows:
        raise ValueError("No edited rows found in manifest.")

    index_file = choose_manifest_value(rows, "index_file", args.index_file)
    n_items = choose_manifest_value(rows, "n_items", args.n_items)
    dataset = WaymoE2E(indexFile=index_file, data_dir=args.data_dir, n_items=n_items)

    bundle = load_sae_bundle(run_root, args.split, args.sae_block, map_location="cpu")
    sae = build_sae_from_checkpoint(bundle["ckpt"], bundle["legacy_norm"]).to(device)
    sae.eval()
    token_tensor, token_key = resolve_token_tensor(bundle["token_blob"], args.sae_block)
    token_name_to_idx = build_token_name_lookup(bundle["token_blob"])
    checkpoint_path = infer_checkpoint_path(run_root, bundle["token_blob"], args.planner_checkpoint)
    planner_model, lit_model = load_model(checkpoint_path, device=device)

    print(f"Loaded {len(rows)} edited manifest rows")
    print(f"Using token key {token_key}")
    print(f"Using planner checkpoint {checkpoint_path}")
    print("Computing baseline SAE stats")
    baseline_stats = compute_baseline_stats(
        sae=sae,
        token_tensor=token_tensor,
        batch_size=args.encode_batch_size,
        device=device,
        min_scale=args.min_scale,
    )

    print("Replaying original/edited pairs")
    replay = replay_pairs(
        rows=rows,
        dataset=dataset,
        planner_model=planner_model,
        lit_model=lit_model,
        sae=sae,
        sae_block=args.sae_block,
        token_blob=bundle["token_blob"],
        token_tensor=token_tensor,
        token_name_to_idx=token_name_to_idx,
        batch_size=args.batch_size,
        device=device,
        allow_name_mismatch=args.allow_name_mismatch,
    )

    pair_csv = output_dir / f"sae_visual_gen_pairs_block_{args.sae_block}.csv"
    feature_delta_pt = output_dir / f"sae_visual_gen_feature_deltas_block_{args.sae_block}.pt"
    summary_csv = output_dir / f"sae_visual_gen_feature_summary_block_{args.sae_block}.csv"
    top_csv = output_dir / f"sae_visual_gen_top_features_block_{args.sae_block}.csv"
    cache_path = output_dir / f"sae_visual_gen_pairs_block_{args.sae_block}.pt"

    write_csv(pair_csv, replay["pair_rows"])
    print(f"Saved pair rows to {pair_csv}")

    print("Saving per-sample feature deltas")
    save_feature_deltas_pt(
        feature_delta_pt,
        rows=rows,
        z_orig=replay["z_orig"],
        z_edit=replay["z_edit"],
        baseline_stats=baseline_stats,
    )
    print(f"Saved feature deltas to {feature_delta_pt}")

    print("Computing feature summaries")
    summary_rows = build_summary_rows(
        rows=rows,
        z_orig=replay["z_orig"],
        z_edit=replay["z_edit"],
        baseline_stats=baseline_stats,
        planner_metrics=replay["planner_metrics"],
    )
    write_csv(summary_csv, summary_rows)
    top_rows = build_top_feature_rows(summary_rows, args.top_k)
    top_fieldnames = ["rank"] + [name for name in top_rows[0].keys() if name != "rank"] if top_rows else None
    write_csv(top_csv, top_rows, fieldnames=top_fieldnames)

    torch.save(
        {
            "manifest": str(manifest_path),
            "run_root": str(run_root),
            "sae_block": args.sae_block,
            "token_key": token_key,
            "token_path": str(bundle["token_path"]),
            "original_source": "token_blob",
            "rows": rows,
            "pair_rows": replay["pair_rows"],
            "tokens_orig": replay["tokens_orig"],
            "tokens_edit": replay["tokens_edit"],
            "z_orig": replay["z_orig"],
            "z_edit": replay["z_edit"],
            "planner_metrics": replay["planner_metrics"],
            "baseline_stats": baseline_stats,
        },
        cache_path,
    )
    print(f"Saved feature summary to {summary_csv}")
    print(f"Saved top features to {top_csv}")
    print(f"Saved tensor cache to {cache_path}")


if __name__ == "__main__":
    main()
