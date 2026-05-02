"""
Analyze SAE features under a paired synthetic stop-sign intervention.

This script selects frames without a detected stop sign in the front camera,
composites a stop-sign PNG into the image, and compares baseline vs patched
SAE activations and predicted trajectories. It ranks features by:

1. Association with synthetic stop-sign presence.
2. Correlation with trajectory shift magnitude.
3. A joint score requiring both presence association and trajectory coupling.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_sae_masked_object_neurons import (
    prepare_model_and_sae,
    run_model_hidden_and_trajectory,
    select_best_trajectory,
)
from analyze_sae_object_neurons import (
    default_index_file,
    load_detection_artifacts,
    resolve_object_label,
)
from loader import WaymoE2E, collate_with_images
from new_sae_utils import get_sae_target_layer
from view_sae_object_analysis import (
    load_front_camera_projection_assets,
    overlay_projected_trajectory,
    projected_visibility_score,
    slot_to_camera_name,
)

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


DEFAULT_PLACEMENTS = (
    {"name": "right_shoulder_near", "x_frac": 0.69, "y_frac": 0.46, "w_frac": 0.14},
    {"name": "right_curb_far", "x_frac": 0.78, "y_frac": 0.42, "w_frac": 0.105},
    {"name": "left_corner_far", "x_frac": 0.13, "y_frac": 0.51, "w_frac": 0.095},
    {"name": "right_lane_edge_mid", "x_frac": 0.63, "y_frac": 0.50, "w_frac": 0.12},
    {"name": "median_far", "x_frac": 0.54, "y_frac": 0.43, "w_frac": 0.09},
)


def detection_record_has_label(
    record: Optional[dict],
    *,
    camera_idx: int,
    label_id: int,
    score_thresh: float,
) -> Optional[bool]:
    if record is None:
        return None
    det = record.get("detections", {}).get(str(camera_idx))
    if det is None:
        return False
    for det_label, det_score in zip(det.get("labels", []), det.get("scores", [])):
        if int(det_label) == int(label_id) and float(det_score) >= score_thresh:
            return True
    return False


def load_stop_sign_asset(path: str | Path, max_side: int = 512) -> torch.Tensor:
    image = Image.open(path).convert("RGBA")
    rgba = np.array(image, dtype=np.uint8)

    alpha = rgba[..., 3]
    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        raise ValueError(f"Stop-sign asset at {path} has no non-transparent pixels")

    top, left = nonzero.min(axis=0)
    bottom, right = nonzero.max(axis=0) + 1
    rgba = rgba[top:bottom, left:right]

    height, width = rgba.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / float(longest)
        resized = Image.fromarray(rgba, mode="RGBA").resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            resample=Image.BILINEAR,
        )
        rgba = np.array(resized, dtype=np.uint8)

    return torch.from_numpy(rgba).permute(2, 0, 1).contiguous()


def resolve_input_path(
    raw_path: str | Path,
    *,
    description: str,
    allow_glob: bool = False,
) -> Path:
    path = Path(raw_path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                SCRIPT_DIR / path,
                REPO_ROOT / path,
            ]
        )

    if allow_glob:
        for candidate in candidates:
            if list(candidate.parent.glob(candidate.name)):
                return candidate
    else:
        for candidate in candidates:
            if candidate.exists():
                return candidate

    locations = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {description} '{raw_path}'. Checked: {locations}")


def load_detection_artifact_metadata(path_pattern: str | Path) -> dict:
    matches = sorted(glob.glob(str(path_pattern)))
    if not matches:
        return {}
    artifact = torch.load(matches[0], map_location="cpu")
    if not isinstance(artifact, dict):
        return {}
    return {
        "split": artifact.get("split"),
        "index_file": artifact.get("index_file"),
        "camera_indices": artifact.get("camera_indices"),
        "score_thresh": artifact.get("score_thresh"),
        "num_records": len(artifact.get("records", [])),
    }


def sample_placement(
    *,
    image_height: int,
    image_width: int,
    pair_idx: int,
    rng: random.Random,
    placement_jitter: float,
    scale_jitter: float,
) -> dict:
    base = DEFAULT_PLACEMENTS[pair_idx % len(DEFAULT_PLACEMENTS)]
    x_frac = base["x_frac"] + rng.uniform(-placement_jitter, placement_jitter)
    y_frac = base["y_frac"] + rng.uniform(-placement_jitter, placement_jitter)
    width_frac = base["w_frac"] * (1.0 + rng.uniform(-scale_jitter, scale_jitter))

    width = max(24, int(round(image_width * width_frac)))
    height = width

    max_x = max(0, image_width - width)
    max_y = max(0, image_height - height)
    x = int(round(x_frac * image_width))
    y = int(round(y_frac * image_height))
    x = max(0, min(max_x, x))
    y = max(0, min(max_y, y))

    return {
        "name": base["name"],
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "image_width": image_width,
        "image_height": image_height,
        "x_frac": x / max(image_width, 1),
        "y_frac": y / max(image_height, 1),
        "w_frac": width / max(image_width, 1),
    }


def composite_rgba_patch(
    image: torch.Tensor,
    asset_rgba: torch.Tensor,
    placement: dict,
) -> torch.Tensor:
    x = int(placement["x"])
    y = int(placement["y"])
    width = int(placement["width"])
    height = int(placement["height"])
    if width <= 0 or height <= 0:
        return image

    device = image.device
    patch = asset_rgba.to(device=device, dtype=torch.float32).unsqueeze(0)
    patch = F.interpolate(patch, size=(height, width), mode="bilinear", align_corners=False)[0]
    rgb = patch[:3]
    alpha = patch[3:4] / 255.0

    out = image.clone()
    region = out[:, y : y + height, x : x + width].to(torch.float32)
    blended = region * (1.0 - alpha) + rgb * alpha
    out[:, y : y + height, x : x + width] = blended.clamp(0, 255).round().to(out.dtype)
    return out


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)


def save_example_figure(
    *,
    output_path: Path,
    frame_name: str,
    placement: dict,
    baseline_image: np.ndarray,
    patched_image: np.ndarray,
    baseline_overlay: np.ndarray,
    patched_overlay: np.ndarray,
    trajectory_shift: float,
    endpoint_shift: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    axes[0].imshow(baseline_image)
    axes[0].set_title(f"Baseline front image\n{frame_name}")
    axes[0].axis("off")

    axes[1].imshow(patched_image)
    axes[1].set_title(
        "Patched front image\n"
        f"{placement['name']} @ ({placement['x']}, {placement['y']}) size={placement['width']}"
    )
    axes[1].axis("off")

    axes[2].imshow(baseline_overlay)
    axes[2].set_title("Baseline projected trajectory")
    axes[2].axis("off")

    axes[3].imshow(patched_overlay)
    axes[3].set_title("Patched projected trajectory")
    axes[3].axis("off")

    fig.suptitle(
        f"traj_shift_l2_mean={trajectory_shift:.4f} | endpoint_shift_l2={endpoint_shift:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def gather_topk_rows(
    metrics: dict,
    *,
    top_k: int,
    sort_key: str,
    min_abs_delta: float,
    positive_only: bool | None,
) -> List[dict]:
    rows = []
    for feature_idx in range(metrics["mean_delta"].numel()):
        mean_delta = float(metrics["mean_delta"][feature_idx].item())
        mean_abs_delta = float(metrics["mean_abs_delta"][feature_idx].item())
        if mean_abs_delta < min_abs_delta:
            continue
        if positive_only is True and mean_delta <= 0.0:
            continue
        if positive_only is False and mean_delta >= 0.0:
            continue

        rows.append(
            {
                "feature_idx": int(feature_idx),
                "presence_corr": float(metrics["presence_corr"][feature_idx].item()),
                "traj_shift_corr": float(metrics["traj_shift_corr"][feature_idx].item()),
                "endpoint_shift_corr": float(metrics["endpoint_shift_corr"][feature_idx].item()),
                "forward_mean_corr": float(metrics["forward_mean_corr"][feature_idx].item()),
                "forward_endpoint_corr": float(metrics["forward_endpoint_corr"][feature_idx].item()),
                "mean_delta": mean_delta,
                "mean_abs_delta": mean_abs_delta,
                "std_delta": float(metrics["std_delta"][feature_idx].item()),
                "positive_consistency": float(metrics["positive_consistency"][feature_idx].item()),
                "negative_consistency": float(metrics["negative_consistency"][feature_idx].item()),
                "baseline_mean": float(metrics["baseline_mean"][feature_idx].item()),
                "patched_mean": float(metrics["patched_mean"][feature_idx].item()),
                "baseline_active_rate": float(metrics["baseline_active_rate"][feature_idx].item()),
                "patched_active_rate": float(metrics["patched_active_rate"][feature_idx].item()),
                "delta_active_rate": float(metrics["delta_active_rate"][feature_idx].item()),
                "presence_score": float(metrics["presence_score"][feature_idx].item()),
                "trajectory_score": float(metrics["trajectory_score"][feature_idx].item()),
                "forward_score": float(metrics["forward_score"][feature_idx].item()),
                "joint_score": float(metrics["joint_score"][feature_idx].item()),
            }
        )

    rows.sort(key=lambda row: row[sort_key], reverse=True)
    return rows[:top_k]


def safe_corr(
    feature_sum: torch.Tensor,
    feature_sumsq: torch.Tensor,
    cross_sum: torch.Tensor,
    other_sum: float,
    other_sumsq: float,
    count: int,
) -> torch.Tensor:
    if count <= 1:
        return torch.zeros_like(feature_sum)

    mean_x = feature_sum / count
    mean_y = other_sum / count
    cov = cross_sum / count - mean_x * mean_y
    var_x = torch.clamp(feature_sumsq / count - mean_x.square(), min=0.0)
    var_y = max(other_sumsq / count - mean_y * mean_y, 0.0)
    if var_y <= 1e-12:
        return torch.zeros_like(feature_sum)
    denom = torch.sqrt(var_x) * math.sqrt(var_y)
    corr = torch.zeros_like(feature_sum)
    valid = denom > 1e-12
    corr[valid] = cov[valid] / denom[valid]
    return corr


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(
    *,
    pair_count: int,
    baseline_sum: torch.Tensor,
    baseline_sumsq: torch.Tensor,
    patched_sum: torch.Tensor,
    patched_sumsq: torch.Tensor,
    baseline_active_sum: torch.Tensor,
    patched_active_sum: torch.Tensor,
    delta_sum: torch.Tensor,
    delta_sumsq: torch.Tensor,
    delta_abs_sum: torch.Tensor,
    delta_positive_count: torch.Tensor,
    delta_negative_count: torch.Tensor,
    traj_cross_sum: torch.Tensor,
    endpoint_cross_sum: torch.Tensor,
    forward_mean_cross_sum: torch.Tensor,
    forward_endpoint_cross_sum: torch.Tensor,
    traj_shift_sum: float,
    traj_shift_sumsq: float,
    endpoint_shift_sum: float,
    endpoint_shift_sumsq: float,
    forward_mean_sum: float,
    forward_mean_sumsq: float,
    forward_endpoint_sum: float,
    forward_endpoint_sumsq: float,
) -> dict:
    baseline_mean = baseline_sum / max(pair_count, 1)
    patched_mean = patched_sum / max(pair_count, 1)
    mean_delta = delta_sum / max(pair_count, 1)
    mean_abs_delta = delta_abs_sum / max(pair_count, 1)
    std_delta = torch.sqrt(torch.clamp(delta_sumsq / max(pair_count, 1) - mean_delta.square(), min=0.0))

    baseline_active_rate = baseline_active_sum / max(pair_count, 1)
    patched_active_rate = patched_active_sum / max(pair_count, 1)
    delta_active_rate = patched_active_rate - baseline_active_rate

    pooled_mean = (baseline_sum + patched_sum) / max(2 * pair_count, 1)
    pooled_var = torch.clamp(
        (baseline_sumsq + patched_sumsq) / max(2 * pair_count, 1) - pooled_mean.square(),
        min=0.0,
    )
    pooled_std = torch.sqrt(pooled_var)
    presence_corr = torch.zeros_like(mean_delta)
    valid = pooled_std > 1e-12
    presence_corr[valid] = 0.5 * (patched_mean[valid] - baseline_mean[valid]) / pooled_std[valid]

    traj_shift_corr = safe_corr(
        feature_sum=delta_sum,
        feature_sumsq=delta_sumsq,
        cross_sum=traj_cross_sum,
        other_sum=traj_shift_sum,
        other_sumsq=traj_shift_sumsq,
        count=pair_count,
    )
    endpoint_shift_corr = safe_corr(
        feature_sum=delta_sum,
        feature_sumsq=delta_sumsq,
        cross_sum=endpoint_cross_sum,
        other_sum=endpoint_shift_sum,
        other_sumsq=endpoint_shift_sumsq,
        count=pair_count,
    )
    forward_mean_corr = safe_corr(
        feature_sum=delta_sum,
        feature_sumsq=delta_sumsq,
        cross_sum=forward_mean_cross_sum,
        other_sum=forward_mean_sum,
        other_sumsq=forward_mean_sumsq,
        count=pair_count,
    )
    forward_endpoint_corr = safe_corr(
        feature_sum=delta_sum,
        feature_sumsq=delta_sumsq,
        cross_sum=forward_endpoint_cross_sum,
        other_sum=forward_endpoint_sum,
        other_sumsq=forward_endpoint_sumsq,
        count=pair_count,
    )

    positive_consistency = delta_positive_count / max(pair_count, 1)
    negative_consistency = delta_negative_count / max(pair_count, 1)

    presence_score = torch.clamp(presence_corr, min=0.0) * mean_abs_delta * positive_consistency
    trajectory_score = torch.clamp(traj_shift_corr, min=0.0) * mean_abs_delta * positive_consistency
    forward_score = torch.clamp(forward_endpoint_corr, min=0.0) * mean_abs_delta * positive_consistency
    joint_score = presence_score * torch.clamp(traj_shift_corr, min=0.0)

    suppressed_presence_score = torch.clamp(-presence_corr, min=0.0) * mean_abs_delta * negative_consistency
    suppressed_trajectory_score = torch.clamp(-traj_shift_corr, min=0.0) * mean_abs_delta * negative_consistency
    suppressed_forward_score = torch.clamp(-forward_endpoint_corr, min=0.0) * mean_abs_delta * negative_consistency
    suppressed_joint_score = suppressed_presence_score * torch.clamp(-traj_shift_corr, min=0.0)

    return {
        "baseline_mean": baseline_mean,
        "patched_mean": patched_mean,
        "mean_delta": mean_delta,
        "mean_abs_delta": mean_abs_delta,
        "std_delta": std_delta,
        "baseline_active_rate": baseline_active_rate,
        "patched_active_rate": patched_active_rate,
        "delta_active_rate": delta_active_rate,
        "presence_corr": presence_corr,
        "traj_shift_corr": traj_shift_corr,
        "endpoint_shift_corr": endpoint_shift_corr,
        "forward_mean_corr": forward_mean_corr,
        "forward_endpoint_corr": forward_endpoint_corr,
        "positive_consistency": positive_consistency,
        "negative_consistency": negative_consistency,
        "presence_score": presence_score,
        "trajectory_score": trajectory_score,
        "forward_score": forward_score,
        "joint_score": joint_score,
        "suppressed_presence_score": suppressed_presence_score,
        "suppressed_trajectory_score": suppressed_trajectory_score,
        "suppressed_forward_score": suppressed_forward_score,
        "suppressed_joint_score": suppressed_joint_score,
    }


def dedupe_feature_rows(*feature_lists: Sequence[dict], max_features: int) -> List[dict]:
    by_feature: Dict[int, dict] = {}
    for rows in feature_lists:
        for row in rows:
            feature_idx = int(row["feature_idx"])
            score = max(
                float(row.get("forward_score", 0.0)),
                float(row.get("trajectory_score", 0.0)),
                float(row.get("joint_score", 0.0)),
            )
            current = by_feature.get(feature_idx)
            if current is None or score > current["_sort_score"]:
                by_feature[feature_idx] = {
                    **row,
                    "_sort_score": score,
                }
    merged = list(by_feature.values())
    merged.sort(key=lambda row: row["_sort_score"], reverse=True)
    for row in merged:
        row.pop("_sort_score", None)
    return merged[:max_features]


@contextlib.contextmanager
def restore_single_feature_hook(
    target_layer,
    sae,
    baseline_hidden: torch.Tensor,
    feature_ids: torch.Tensor,
):
    def replace_with_restored_feature(module, inputs, output):
        del module, inputs
        flat_output = output.reshape(-1, output.size(-1))
        hidden, preprocess_stats = sae.encode(
            flat_output,
            return_preprocess_stats=True,
        )
        hidden = hidden.clone()
        row_ids = torch.arange(hidden.size(0), device=hidden.device)
        hidden[row_ids, feature_ids.to(device=hidden.device)] = baseline_hidden.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )[row_ids, feature_ids.to(device=hidden.device)]
        reconstructed = sae.decode_to_input(hidden, preprocess_stats=preprocess_stats)
        return reconstructed.view_as(output)

    handle = target_layer.register_forward_hook(replace_with_restored_feature)
    try:
        yield
    finally:
        handle.remove()


def run_single_feature_restorations(
    *,
    model,
    sae,
    target_layer,
    past: torch.Tensor,
    intent: torch.Tensor,
    patched_images: Sequence[torch.Tensor],
    baseline_hidden: torch.Tensor,
    feature_ids: Sequence[int],
    chunk_size: int,
) -> Dict[int, torch.Tensor]:
    if not feature_ids:
        return {}

    batch_size = past.size(0)
    outputs: Dict[int, torch.Tensor] = {}
    for start in range(0, len(feature_ids), chunk_size):
        feature_chunk = list(feature_ids[start : start + chunk_size])
        n_features = len(feature_chunk)
        repeated_past = past.repeat(n_features, 1, 1)
        repeated_intent = intent.repeat(n_features)
        repeated_images = [camera_batch.repeat(n_features, 1, 1, 1) for camera_batch in patched_images]
        repeated_hidden = baseline_hidden.repeat(n_features, 1)
        repeated_feature_ids = torch.repeat_interleave(
            torch.as_tensor(feature_chunk, dtype=torch.long, device=past.device),
            repeats=batch_size,
        )

        with torch.no_grad():
            with restore_single_feature_hook(
                target_layer,
                sae,
                baseline_hidden=repeated_hidden,
                feature_ids=repeated_feature_ids,
            ):
                restored_output = model(
                    {
                        "PAST": repeated_past,
                        "IMAGES": repeated_images,
                        "INTENT": repeated_intent,
                    }
                )

        restored_traj = restored_output["trajectory"]
        restored_scores = restored_output.get("scores")
        restored_best = restored_traj
        if isinstance(restored_output, dict):
            restored_best = select_best_trajectory(
                {
                    "trajectory": restored_traj,
                    "scores": restored_scores,
                }
            )
        restored_best = restored_best.detach().cpu()
        restored_best = restored_best.reshape(
            n_features,
            batch_size,
            restored_best.size(-2),
            restored_best.size(-1),
        )
        for feature_offset, feature_idx in enumerate(feature_chunk):
            outputs[int(feature_idx)] = restored_best[feature_offset]
    return outputs


def evaluate_causal_feature_responsibility(
    *,
    dataset: WaymoE2E,
    model,
    sae,
    target_layer,
    stop_sign_asset: torch.Tensor,
    pair_rows: Sequence[dict],
    feature_rows: Sequence[dict],
    front_camera_idx: int,
    device: torch.device,
    batch_size: int,
    feature_chunk_size: int,
) -> dict:
    if not feature_rows:
        return {
            "evaluated_pair_count": 0,
            "evaluated_feature_count": 0,
            "top_single_feature_responsibility": [],
            "top_feature_set_additive_proxy": [],
        }

    feature_ids = [int(row["feature_idx"]) for row in feature_rows]
    feature_meta = {int(row["feature_idx"]): row for row in feature_rows}
    stats = {
        feature_idx: {
            "count": 0,
            "endpoint_frac_count": 0,
            "mean_frac_count": 0,
            "forward_endpoint_rescue_sum": 0.0,
            "forward_mean_rescue_sum": 0.0,
            "forward_endpoint_rescue_frac_sum": 0.0,
            "forward_mean_rescue_frac_sum": 0.0,
            "trajectory_l2_reduction_sum": 0.0,
            "positive_endpoint_rescue_count": 0,
            "positive_mean_rescue_count": 0,
        }
        for feature_idx in feature_ids
    }
    evaluated_pair_count = 0

    selected_rows = list(pair_rows)
    for start in range(0, len(selected_rows), batch_size):
        row_batch = selected_rows[start : start + batch_size]
        if not row_batch:
            continue
        samples = [dict(dataset[int(row["dataset_idx"])]) for row in row_batch]
        batch = collate_with_images(samples)
        past = batch["PAST"].to(device)
        intent = batch["INTENT"].to(device)
        images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
        patched_images = [camera_batch.clone() for camera_batch in images]
        for local_idx, row in enumerate(row_batch):
            patched_images[front_camera_idx][local_idx] = composite_rgba_patch(
                patched_images[front_camera_idx][local_idx],
                stop_sign_asset,
                row["placement"],
            )

        baseline_hidden, baseline_traj = run_model_hidden_and_trajectory(
            model=model,
            sae=sae,
            past=past,
            intent=intent,
            images=images,
        )
        _, patched_traj = run_model_hidden_and_trajectory(
            model=model,
            sae=sae,
            past=past,
            intent=intent,
            images=patched_images,
        )
        baseline_hidden = baseline_hidden.to(device=device, dtype=torch.float32)
        baseline_traj = baseline_traj.to(torch.float64)
        patched_traj = patched_traj.to(torch.float64)

        baseline_endpoint_drop = baseline_traj[:, -1, 0] - patched_traj[:, -1, 0]
        baseline_mean_drop = baseline_traj[..., 0].mean(dim=-1) - patched_traj[..., 0].mean(dim=-1)
        baseline_l2_shift = torch.norm(baseline_traj - patched_traj, dim=-1).mean(dim=-1)

        restored_by_feature = run_single_feature_restorations(
            model=model,
            sae=sae,
            target_layer=target_layer,
            past=past,
            intent=intent,
            patched_images=patched_images,
            baseline_hidden=baseline_hidden,
            feature_ids=feature_ids,
            chunk_size=feature_chunk_size,
        )

        for feature_idx, restored_traj in restored_by_feature.items():
            restored_traj = restored_traj.to(torch.float64)
            endpoint_rescue = restored_traj[:, -1, 0] - patched_traj[:, -1, 0]
            mean_rescue = restored_traj[..., 0].mean(dim=-1) - patched_traj[..., 0].mean(dim=-1)
            restored_l2_shift = torch.norm(baseline_traj - restored_traj, dim=-1).mean(dim=-1)
            l2_reduction = baseline_l2_shift - restored_l2_shift

            valid_endpoint = baseline_endpoint_drop > 1e-6
            valid_mean = baseline_mean_drop > 1e-6
            feature_stat = stats[feature_idx]
            feature_stat["count"] += endpoint_rescue.numel()
            feature_stat["forward_endpoint_rescue_sum"] += float(endpoint_rescue.sum().item())
            feature_stat["forward_mean_rescue_sum"] += float(mean_rescue.sum().item())
            feature_stat["trajectory_l2_reduction_sum"] += float(l2_reduction.sum().item())
            feature_stat["positive_endpoint_rescue_count"] += int((endpoint_rescue > 0).sum().item())
            feature_stat["positive_mean_rescue_count"] += int((mean_rescue > 0).sum().item())
            if valid_endpoint.any():
                feature_stat["endpoint_frac_count"] += int(valid_endpoint.sum().item())
                feature_stat["forward_endpoint_rescue_frac_sum"] += float(
                    (endpoint_rescue[valid_endpoint] / baseline_endpoint_drop[valid_endpoint]).sum().item()
                )
            if valid_mean.any():
                feature_stat["mean_frac_count"] += int(valid_mean.sum().item())
                feature_stat["forward_mean_rescue_frac_sum"] += float(
                    (mean_rescue[valid_mean] / baseline_mean_drop[valid_mean]).sum().item()
                )

        evaluated_pair_count += len(row_batch)

    rows = []
    for feature_idx in feature_ids:
        feature_stat = stats[feature_idx]
        count = max(feature_stat["count"], 1)
        endpoint_frac_count = max(feature_stat["endpoint_frac_count"], 1)
        mean_frac_count = max(feature_stat["mean_frac_count"], 1)
        meta = feature_meta[feature_idx]
        rows.append(
            {
                "feature_idx": feature_idx,
                "mean_delta": float(meta["mean_delta"]),
                "forward_score": float(meta.get("forward_score", 0.0)),
                "trajectory_score": float(meta.get("trajectory_score", 0.0)),
                "joint_score": float(meta.get("joint_score", 0.0)),
                "mean_forward_endpoint_rescue_m": feature_stat["forward_endpoint_rescue_sum"] / count,
                "mean_forward_mean_rescue_m": feature_stat["forward_mean_rescue_sum"] / count,
                "mean_forward_endpoint_rescue_fraction": feature_stat["forward_endpoint_rescue_frac_sum"] / endpoint_frac_count,
                "mean_forward_mean_rescue_fraction": feature_stat["forward_mean_rescue_frac_sum"] / mean_frac_count,
                "mean_trajectory_l2_reduction": feature_stat["trajectory_l2_reduction_sum"] / count,
                "positive_endpoint_rescue_rate": feature_stat["positive_endpoint_rescue_count"] / count,
                "positive_mean_rescue_rate": feature_stat["positive_mean_rescue_count"] / count,
            }
        )
    rows.sort(
        key=lambda row: (
            row["mean_forward_endpoint_rescue_m"],
            row["mean_forward_endpoint_rescue_fraction"],
            row["mean_trajectory_l2_reduction"],
        ),
        reverse=True,
    )

    additive_proxy_rows = []
    running_feature_ids: List[int] = []
    for row in rows[: min(10, len(rows))]:
        running_feature_ids.append(int(row["feature_idx"]))
        additive_proxy_rows.append(
            {
                "top_n_features": len(running_feature_ids),
                "feature_ids": list(running_feature_ids),
                "approx_mean_single_feature_endpoint_rescue_m": float(
                    sum(
                        next(
                            candidate["mean_forward_endpoint_rescue_m"]
                            for candidate in rows
                            if int(candidate["feature_idx"]) == feature_idx
                        )
                        for feature_idx in running_feature_ids
                    )
                ),
            }
        )

    return {
        "evaluated_pair_count": evaluated_pair_count,
        "evaluated_feature_count": len(feature_ids),
        "top_single_feature_responsibility": rows[:25],
        "top_feature_set_additive_proxy": additive_proxy_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, required=True, help="Glob or path for saved detection artifacts")
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="SAE checkpoint .pt, extracted directory, or sae_checkpoints.tar.gz archive")
    parser.add_argument("--data_dir", type=str, required=True, help="Waymo dataset directory")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stop_sign_asset", type=str, default="stop sign.png")
    parser.add_argument("--object_name", type=str, default="stop sign")
    parser.add_argument("--label_id", type=int, default=None)
    parser.add_argument("--model_checkpoint_path", default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt", type=str)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--index_file", type=str, default=None, help="Override split index file")
    parser.add_argument("--n_items", type=int, default=None, help="Maximum raw frames to scan")
    parser.add_argument("--num_pairs", type=int, default=10000, help="Number of clean/pasted pairs to analyze")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=3, help="Transformer block index whose output query state is SAE-modeled")
    parser.add_argument("--score_thresh", type=float, default=0.4)
    parser.add_argument("--front_camera_idx", type=int, default=1, help="Dataset camera slot used for patching and stop-sign filtering")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--min_abs_delta", type=float, default=1e-4, help="Minimum mean absolute delta for ranked features")
    parser.add_argument("--placement_jitter", type=float, default=0.02, help="Uniform jitter added to placement x/y fractions")
    parser.add_argument("--scale_jitter", type=float, default=0.10, help="Uniform jitter applied to stop-sign width fraction")
    parser.add_argument("--asset_max_side", type=int, default=512, help="Pre-resize the stop-sign asset for faster interpolation")
    parser.add_argument("--num_visualizations", type=int, default=8, help="Number of highest-shift examples to visualize")
    parser.add_argument("--causal_top_candidates", type=int, default=32, help="Number of SAE features to evaluate with single-feature causal restoration")
    parser.add_argument("--causal_max_pairs", type=int, default=256, help="Maximum clean/pasted pairs to use for the causal restoration pass")
    parser.add_argument("--causal_feature_chunk_size", type=int, default=8, help="How many feature-specific interventions to batch into one forward pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    detections_path = resolve_input_path(
        args.detections,
        description="detection artifact path",
        allow_glob=True,
    )
    model_checkpoint_path = resolve_input_path(
        args.model_checkpoint_path,
        description="model checkpoint path",
    )
    sae_checkpoint_path = resolve_input_path(
        args.sae_checkpoint_path,
        description="SAE checkpoint path",
    )
    stop_sign_asset_path = resolve_input_path(
        args.stop_sign_asset,
        description="stop-sign asset path",
    )
    data_dir = resolve_input_path(args.data_dir, description="dataset directory")
    index_file = resolve_input_path(
        args.index_file or default_index_file(args.split),
        description="dataset index file",
    )

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    categories, frame_to_record = load_detection_artifacts(str(detections_path))
    detection_metadata = load_detection_artifact_metadata(str(detections_path))
    stop_sign_label_id, stop_sign_label_name = resolve_object_label(
        categories,
        args.object_name,
        args.label_id,
    )

    model, sae, dict_size = prepare_model_and_sae(
        model_checkpoint_path=str(model_checkpoint_path),
        sae_checkpoint_path=str(sae_checkpoint_path),
        block_idx=args.block_idx,
        device=device,
    )

    stop_sign_asset = load_stop_sign_asset(stop_sign_asset_path, max_side=args.asset_max_side)

    dataset = WaymoE2E(indexFile=str(index_file), data_dir=str(data_dir), n_items=None)
    if args.n_items is not None:
        dataset.indexes = dataset.indexes[: args.n_items]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    baseline_sum = torch.zeros(dict_size, dtype=torch.float64)
    baseline_sumsq = torch.zeros(dict_size, dtype=torch.float64)
    patched_sum = torch.zeros(dict_size, dtype=torch.float64)
    patched_sumsq = torch.zeros(dict_size, dtype=torch.float64)
    baseline_active_sum = torch.zeros(dict_size, dtype=torch.float64)
    patched_active_sum = torch.zeros(dict_size, dtype=torch.float64)
    delta_sum = torch.zeros(dict_size, dtype=torch.float64)
    delta_sumsq = torch.zeros(dict_size, dtype=torch.float64)
    delta_abs_sum = torch.zeros(dict_size, dtype=torch.float64)
    delta_positive_count = torch.zeros(dict_size, dtype=torch.float64)
    delta_negative_count = torch.zeros(dict_size, dtype=torch.float64)
    traj_cross_sum = torch.zeros(dict_size, dtype=torch.float64)
    endpoint_cross_sum = torch.zeros(dict_size, dtype=torch.float64)
    forward_mean_cross_sum = torch.zeros(dict_size, dtype=torch.float64)
    forward_endpoint_cross_sum = torch.zeros(dict_size, dtype=torch.float64)

    pair_count = 0
    raw_frames_scanned = 0
    missing_detection_frames = 0
    skipped_existing_stop_sign = 0
    traj_shift_sum = 0.0
    traj_shift_sumsq = 0.0
    endpoint_shift_sum = 0.0
    endpoint_shift_sumsq = 0.0
    forward_mean_sum = 0.0
    forward_mean_sumsq = 0.0
    forward_endpoint_sum = 0.0
    forward_endpoint_sumsq = 0.0
    pair_rows: List[dict] = []
    skipped_missing_front_camera = 0
    running_dataset_idx = 0
    target_layer = get_sae_target_layer(model, args.block_idx)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Pasted stop-sign SAE analysis"):
            if pair_count >= args.num_pairs:
                break

            frame_names = batch["NAME"]
            batch_size = len(frame_names)
            raw_frames_scanned += batch_size

            accepted_indices: List[int] = []
            accepted_rows: List[dict] = []

            for sample_idx, frame_name in enumerate(frame_names):
                if pair_count + len(accepted_indices) >= args.num_pairs:
                    break

                has_stop_sign = detection_record_has_label(
                    frame_to_record.get(frame_name),
                    camera_idx=args.front_camera_idx,
                    label_id=stop_sign_label_id,
                    score_thresh=args.score_thresh,
                )
                if has_stop_sign is None:
                    missing_detection_frames += 1
                    continue
                if has_stop_sign:
                    skipped_existing_stop_sign += 1
                    continue

                front_images_jpeg = batch["IMAGES_JPEG"][sample_idx]
                if args.front_camera_idx >= len(front_images_jpeg):
                    skipped_missing_front_camera += 1
                    continue

                accepted_indices.append(sample_idx)
                accepted_rows.append(
                    {
                        "dataset_idx": running_dataset_idx + sample_idx,
                        "frame_name": frame_name,
                    }
                )

            if not accepted_indices:
                running_dataset_idx += batch_size
                continue

            past = batch["PAST"][accepted_indices].to(device)
            intent = batch["INTENT"][accepted_indices].to(device)
            images = model.decode_batch_jpeg(
                [batch["IMAGES_JPEG"][sample_idx] for sample_idx in accepted_indices],
                device=device,
            )
            patched_images = [cam.clone() for cam in images]

            for local_idx, row in enumerate(accepted_rows):
                front_image = patched_images[args.front_camera_idx][local_idx]
                placement = sample_placement(
                    image_height=int(front_image.shape[-2]),
                    image_width=int(front_image.shape[-1]),
                    pair_idx=pair_count + local_idx,
                    rng=rng,
                    placement_jitter=args.placement_jitter,
                    scale_jitter=args.scale_jitter,
                )
                patched_images[args.front_camera_idx][local_idx] = composite_rgba_patch(
                    front_image,
                    stop_sign_asset,
                    placement,
                )
                row["placement"] = placement

            baseline_hidden, baseline_traj = run_model_hidden_and_trajectory(
                model=model,
                sae=sae,
                past=past,
                intent=intent,
                images=images,
            )
            patched_hidden, patched_traj = run_model_hidden_and_trajectory(
                model=model,
                sae=sae,
                past=past,
                intent=intent,
                images=patched_images,
            )

            traj_shift = torch.norm(baseline_traj - patched_traj, dim=-1).mean(dim=-1).to(torch.float64)
            endpoint_shift = torch.norm(baseline_traj[:, -1] - patched_traj[:, -1], dim=-1).to(torch.float64)
            forward_mean_drop = (baseline_traj[..., 0] - patched_traj[..., 0]).mean(dim=-1).to(torch.float64)
            forward_endpoint_drop = (baseline_traj[:, -1, 0] - patched_traj[:, -1, 0]).to(torch.float64)
            delta = (patched_hidden - baseline_hidden).to(torch.float64)

            baseline_sum += baseline_hidden.sum(dim=0)
            baseline_sumsq += baseline_hidden.square().sum(dim=0)
            patched_sum += patched_hidden.sum(dim=0)
            patched_sumsq += patched_hidden.square().sum(dim=0)
            baseline_active_sum += (baseline_hidden > 0).to(torch.float64).sum(dim=0)
            patched_active_sum += (patched_hidden > 0).to(torch.float64).sum(dim=0)
            delta_sum += delta.sum(dim=0)
            delta_sumsq += delta.square().sum(dim=0)
            delta_abs_sum += delta.abs().sum(dim=0)
            delta_positive_count += (delta > 0).to(torch.float64).sum(dim=0)
            delta_negative_count += (delta < 0).to(torch.float64).sum(dim=0)
            traj_cross_sum += (delta * traj_shift.unsqueeze(1)).sum(dim=0)
            endpoint_cross_sum += (delta * endpoint_shift.unsqueeze(1)).sum(dim=0)
            forward_mean_cross_sum += (delta * forward_mean_drop.unsqueeze(1)).sum(dim=0)
            forward_endpoint_cross_sum += (delta * forward_endpoint_drop.unsqueeze(1)).sum(dim=0)

            traj_shift_sum += float(traj_shift.sum().item())
            traj_shift_sumsq += float(traj_shift.square().sum().item())
            endpoint_shift_sum += float(endpoint_shift.sum().item())
            endpoint_shift_sumsq += float(endpoint_shift.square().sum().item())
            forward_mean_sum += float(forward_mean_drop.sum().item())
            forward_mean_sumsq += float(forward_mean_drop.square().sum().item())
            forward_endpoint_sum += float(forward_endpoint_drop.sum().item())
            forward_endpoint_sumsq += float(forward_endpoint_drop.square().sum().item())

            for local_idx, row in enumerate(accepted_rows):
                pair_rows.append(
                    {
                        "dataset_idx": int(row["dataset_idx"]),
                        "frame_name": row["frame_name"],
                        "placement": row["placement"],
                        "trajectory_shift_l2_mean": float(traj_shift[local_idx].item()),
                        "trajectory_endpoint_shift_l2": float(endpoint_shift[local_idx].item()),
                        "trajectory_forward_mean_drop_m": float(forward_mean_drop[local_idx].item()),
                        "trajectory_forward_endpoint_drop_m": float(forward_endpoint_drop[local_idx].item()),
                        "visibility_score": None,
                    }
                )

            pair_count += len(accepted_rows)
            running_dataset_idx += batch_size

    if pair_count == 0:
        if missing_detection_frames == raw_frames_scanned and raw_frames_scanned > 0:
            raise RuntimeError(
                "No dataset frame names overlapped with the detection artifact. "
                f"Requested split={args.split} index_file={index_file}, but the detection artifact metadata says "
                f"split={detection_metadata.get('split')} index_file={detection_metadata.get('index_file')}. "
                "Make sure --split/--index_file match the detections file you are replaying."
            )
        raise RuntimeError("No clean front-camera frames were found to paste a synthetic stop sign into")

    metrics = build_metrics(
        pair_count=pair_count,
        baseline_sum=baseline_sum,
        baseline_sumsq=baseline_sumsq,
        patched_sum=patched_sum,
        patched_sumsq=patched_sumsq,
        baseline_active_sum=baseline_active_sum,
        patched_active_sum=patched_active_sum,
        delta_sum=delta_sum,
        delta_sumsq=delta_sumsq,
        delta_abs_sum=delta_abs_sum,
        delta_positive_count=delta_positive_count,
        delta_negative_count=delta_negative_count,
        traj_cross_sum=traj_cross_sum,
        endpoint_cross_sum=endpoint_cross_sum,
        forward_mean_cross_sum=forward_mean_cross_sum,
        forward_endpoint_cross_sum=forward_endpoint_cross_sum,
        traj_shift_sum=traj_shift_sum,
        traj_shift_sumsq=traj_shift_sumsq,
        endpoint_shift_sum=endpoint_shift_sum,
        endpoint_shift_sumsq=endpoint_shift_sumsq,
        forward_mean_sum=forward_mean_sum,
        forward_mean_sumsq=forward_mean_sumsq,
        forward_endpoint_sum=forward_endpoint_sum,
        forward_endpoint_sumsq=forward_endpoint_sumsq,
    )

    top_presence_positive = gather_topk_rows(
        metrics,
        top_k=args.top_k,
        sort_key="presence_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=True,
    )
    top_presence_suppressed = gather_topk_rows(
        {
            **metrics,
            "presence_score": metrics["suppressed_presence_score"],
            "trajectory_score": metrics["suppressed_trajectory_score"],
            "forward_score": metrics["suppressed_forward_score"],
            "joint_score": metrics["suppressed_joint_score"],
        },
        top_k=args.top_k,
        sort_key="presence_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=False,
    )
    top_traj_positive = gather_topk_rows(
        metrics,
        top_k=args.top_k,
        sort_key="trajectory_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=True,
    )
    top_traj_suppressed = gather_topk_rows(
        {
            **metrics,
            "presence_score": metrics["suppressed_presence_score"],
            "trajectory_score": metrics["suppressed_trajectory_score"],
            "forward_score": metrics["suppressed_forward_score"],
            "joint_score": metrics["suppressed_joint_score"],
        },
        top_k=args.top_k,
        sort_key="trajectory_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=False,
    )
    top_forward_positive = gather_topk_rows(
        metrics,
        top_k=args.top_k,
        sort_key="forward_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=True,
    )
    top_forward_suppressed = gather_topk_rows(
        {
            **metrics,
            "presence_score": metrics["suppressed_presence_score"],
            "trajectory_score": metrics["suppressed_trajectory_score"],
            "forward_score": metrics["suppressed_forward_score"],
            "joint_score": metrics["suppressed_joint_score"],
        },
        top_k=args.top_k,
        sort_key="forward_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=False,
    )
    top_joint_positive = gather_topk_rows(
        metrics,
        top_k=args.top_k,
        sort_key="joint_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=True,
    )
    top_joint_suppressed = gather_topk_rows(
        {
            **metrics,
            "presence_score": metrics["suppressed_presence_score"],
            "trajectory_score": metrics["suppressed_trajectory_score"],
            "forward_score": metrics["suppressed_forward_score"],
            "joint_score": metrics["suppressed_joint_score"],
        },
        top_k=args.top_k,
        sort_key="joint_score",
        min_abs_delta=args.min_abs_delta,
        positive_only=False,
    )

    pair_rows.sort(
        key=lambda row: (
            row["trajectory_forward_endpoint_drop_m"],
            row["trajectory_shift_l2_mean"],
        ),
        reverse=True,
    )

    causal_candidates = dedupe_feature_rows(
        top_forward_positive,
        top_forward_suppressed,
        top_joint_positive,
        top_joint_suppressed,
        max_features=args.causal_top_candidates,
    )
    causal_rows = [
        row for row in pair_rows
        if row["trajectory_forward_endpoint_drop_m"] > 1e-6
    ][: args.causal_max_pairs]
    causal_summary = evaluate_causal_feature_responsibility(
        dataset=dataset,
        model=model,
        sae=sae,
        target_layer=target_layer,
        stop_sign_asset=stop_sign_asset,
        pair_rows=causal_rows,
        feature_rows=causal_candidates,
        front_camera_idx=args.front_camera_idx,
        device=device,
        batch_size=args.batch_size,
        feature_chunk_size=args.causal_feature_chunk_size,
    )

    output = {
        "method": (
            "paired synthetic stop-sign intervention; SAE hidden = "
            "sae.encode(model.blocks[block_idx](query, tokens) output) with the topk_aux SAE checkpoint"
        ),
        "important_caveat": (
            "This is a synthetic overlay study. The correlational rankings surface "
            "candidate stop-sign-responsive features, while the causal table measures "
            "how much restoring one feature at a time rescues the patched trajectory."
        ),
        "pair_count": pair_count,
        "raw_frames_scanned": raw_frames_scanned,
        "missing_detection_frames": missing_detection_frames,
        "skipped_existing_stop_sign": skipped_existing_stop_sign,
        "skipped_missing_front_camera": skipped_missing_front_camera,
        "front_camera_idx": args.front_camera_idx,
        "object_label_id": stop_sign_label_id,
        "object_label_name": stop_sign_label_name,
        "score_thresh": args.score_thresh,
        "split": args.split,
        "seed": args.seed,
        "top_k": args.top_k,
        "min_abs_delta": args.min_abs_delta,
        "sae_feature_count": dict_size,
        "detections": str(detections_path),
        "detection_artifact_metadata": detection_metadata,
        "sae_checkpoint_path": str(sae_checkpoint_path),
        "resolved_sae_checkpoint_path": getattr(sae, "_resolved_checkpoint_path", str(sae_checkpoint_path)),
        "model_checkpoint_path": str(model_checkpoint_path),
        "data_dir": str(data_dir),
        "index_file": str(index_file),
        "stop_sign_asset": str(stop_sign_asset_path),
        "mean_trajectory_shift_l2": traj_shift_sum / pair_count,
        "mean_endpoint_shift_l2": endpoint_shift_sum / pair_count,
        "mean_forward_trajectory_drop_m": forward_mean_sum / pair_count,
        "mean_forward_endpoint_drop_m": forward_endpoint_sum / pair_count,
        "ranking_definitions": {
            "presence_score": "max(presence_corr, 0) * mean_abs_delta * positive_consistency",
            "trajectory_score": "max(traj_shift_corr, 0) * mean_abs_delta * positive_consistency",
            "forward_score": "max(forward_endpoint_corr, 0) * mean_abs_delta * positive_consistency",
            "joint_score": "presence_score * max(traj_shift_corr, 0)",
            "suppressed_scores": "same definitions but using negated correlations and negative_consistency",
        },
        "top_presence_positive_features": top_presence_positive,
        "top_presence_suppressed_features": top_presence_suppressed,
        "top_trajectory_positive_features": top_traj_positive,
        "top_trajectory_suppressed_features": top_traj_suppressed,
        "top_forward_positive_features": top_forward_positive,
        "top_forward_suppressed_features": top_forward_suppressed,
        "top_joint_positive_features": top_joint_positive,
        "top_joint_suppressed_features": top_joint_suppressed,
        "causal_candidates": causal_candidates,
        "causal_single_feature_responsibility": causal_summary,
        "top_pairs_by_forward_endpoint_drop": pair_rows[: max(args.num_visualizations * 4, args.top_k)],
        "top_pairs_by_trajectory_shift": sorted(
            pair_rows,
            key=lambda row: row["trajectory_shift_l2_mean"],
            reverse=True,
        )[: max(args.num_visualizations * 4, args.top_k)],
    }

    write_csv(output_dir / "top_joint_positive_features.csv", top_joint_positive)
    write_csv(output_dir / "top_joint_suppressed_features.csv", top_joint_suppressed)
    write_csv(output_dir / "top_presence_positive_features.csv", top_presence_positive)
    write_csv(output_dir / "top_presence_suppressed_features.csv", top_presence_suppressed)
    write_csv(output_dir / "top_trajectory_positive_features.csv", top_traj_positive)
    write_csv(output_dir / "top_trajectory_suppressed_features.csv", top_traj_suppressed)
    write_csv(output_dir / "top_forward_positive_features.csv", top_forward_positive)
    write_csv(output_dir / "top_forward_suppressed_features.csv", top_forward_suppressed)
    write_csv(
        output_dir / "causal_single_feature_responsibility.csv",
        causal_summary["top_single_feature_responsibility"],
    )

    if args.num_visualizations > 0:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        visualized = 0
        skipped_for_visibility = 0
        try:
            display_camera_name = slot_to_camera_name(args.front_camera_idx)
        except ValueError as exc:
            print(f"Skipping visualization stage: {exc}")
        else:
            for row in pair_rows:
                if visualized >= args.num_visualizations:
                    break

                dataset_idx = int(row["dataset_idx"])
                sample = dataset[dataset_idx]
                batch = collate_with_images([dict(sample)])
                images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
                patched_images = [cam.clone() for cam in images]
                patched_images[args.front_camera_idx][0] = composite_rgba_patch(
                    patched_images[args.front_camera_idx][0],
                    stop_sign_asset,
                    row["placement"],
                )

                baseline_hidden, baseline_traj = run_model_hidden_and_trajectory(
                    model=model,
                    sae=sae,
                    past=batch["PAST"].to(device),
                    intent=batch["INTENT"].to(device),
                    images=images,
                )
                patched_hidden, patched_traj = run_model_hidden_and_trajectory(
                    model=model,
                    sae=sae,
                    past=batch["PAST"].to(device),
                    intent=batch["INTENT"].to(device),
                    images=patched_images,
                )
                del baseline_hidden, patched_hidden

                baseline_front_image, calibration = load_front_camera_projection_assets(
                    dataset,
                    dataset_idx,
                    camera_name=display_camera_name,
                )
                patched_front_image = tensor_image_to_numpy(patched_images[args.front_camera_idx][0])
                visibility_score = projected_visibility_score(
                    calibration=calibration,
                    trajectory_xy=baseline_traj[0],
                    future_xy=batch["FUTURE"][0],
                )
                row["visibility_score"] = visibility_score
                if visibility_score < 2:
                    skipped_for_visibility += 1
                    continue

                baseline_overlay = overlay_projected_trajectory(
                    image=baseline_front_image,
                    calibration=calibration,
                    trajectory_xy=baseline_traj[0],
                    future_xy=batch["FUTURE"][0],
                    pred_color=(255, 0, 0),
                )
                patched_overlay = overlay_projected_trajectory(
                    image=patched_front_image,
                    calibration=calibration,
                    trajectory_xy=patched_traj[0],
                    future_xy=batch["FUTURE"][0],
                    pred_color=(255, 165, 0),
                )

                save_example_figure(
                    output_path=viz_dir / f"example_{visualized + 1:02d}.png",
                    frame_name=row["frame_name"],
                    placement=row["placement"],
                    baseline_image=baseline_front_image,
                    patched_image=patched_front_image,
                    baseline_overlay=baseline_overlay,
                    patched_overlay=patched_overlay,
                    trajectory_shift=row["trajectory_shift_l2_mean"],
                    endpoint_shift=row["trajectory_endpoint_shift_l2"],
                )
                visualized += 1

        output["visualized_examples"] = visualized
        output["skipped_visualizations_for_low_visibility"] = skipped_for_visibility

    (output_dir / "stop_sign_sae_paired_analysis.json").write_text(json.dumps(output, indent=2))
    write_csv(output_dir / "all_pairs.csv", pair_rows)

    print(
        f"Analyzed {pair_count} clean/pasted stop-sign pairs "
        f"(scanned {raw_frames_scanned} raw frames); "
        f"mean trajectory shift={traj_shift_sum / pair_count:.6f}; "
        f"mean forward endpoint drop={forward_endpoint_sum / pair_count:.6f} m"
    )
    print(f"Saved paired stop-sign SAE analysis to {output_dir}")


if __name__ == "__main__":
    main()
