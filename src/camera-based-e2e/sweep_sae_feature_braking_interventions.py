"""
Sweep top SAE feature interventions and measure induced braking.

This script is meant to follow ``analyze_sae_pasted_stop_sign_neurons.py``.
It loads top candidate features from the analysis JSON, applies stronger
interventions to clean or object-present frames, and reports which feature/set
and intervention strength causes the largest forward-trajectory decrease.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from analyze_sae_masked_object_neurons import (
    prepare_model_and_sae,
    run_model_hidden_and_trajectory,
    select_best_trajectory,
)
from analyze_sae_object_neurons import (
    default_index_file,
    frame_has_object,
    load_detection_artifacts,
    past_current_speed,
    resolve_object_label,
)
from loader import WaymoE2E, collate_with_images
from models.sae import SparseAutoencoder
from new_sae_utils import get_sae_target_layer

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def resolve_input_path(
    raw_path: str | Path,
    *,
    description: str,
) -> Path:
    path = Path(raw_path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, SCRIPT_DIR / path, REPO_ROOT / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    locations = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {description} '{raw_path}'. Checked: {locations}")


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_ints(raw: Optional[str]) -> list[int]:
    if not raw:
        return []
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def parse_csv_floats(raw: Optional[str]) -> list[float]:
    if not raw:
        return []
    return [float(piece.strip()) for piece in raw.split(",") if piece.strip()]


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_nested(data: dict, key_path: str):
    value = data
    for part in key_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def choose_feature_rows(analysis: dict, feature_key: str, feature_limit: int) -> list[dict]:
    if feature_key != "auto":
        rows = get_nested(analysis, feature_key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Analysis JSON has no feature rows under '{feature_key}'")
        return rows[:feature_limit]

    for candidate_key in (
        "causal_single_feature_responsibility.top_single_feature_responsibility",
        "causal_candidates",
        "top_forward_positive_features",
        "top_joint_positive_features",
        "top_trajectory_positive_features",
        "top_presence_positive_features",
    ):
        rows = get_nested(analysis, candidate_key)
        if isinstance(rows, list) and rows:
            return rows[:feature_limit]
    raise ValueError("Could not find any feature list in the analysis JSON")


def choose_feature_groups(
    feature_rows: Sequence[dict],
    *,
    feature_counts: Sequence[int],
    include_single_feature_sweep: bool,
) -> list[dict]:
    feature_ids = [int(row["feature_idx"]) for row in feature_rows]
    groups = []
    seen = set()

    if include_single_feature_sweep:
        for feature_id in feature_ids:
            key = ("single", (feature_id,))
            if key in seen:
                continue
            seen.add(key)
            groups.append(
                {
                    "group_name": f"single_feature_{feature_id}",
                    "feature_ids": [feature_id],
                    "group_type": "single",
                }
            )

    for count in feature_counts:
        if count <= 0:
            continue
        prefix = tuple(feature_ids[: min(count, len(feature_ids))])
        if not prefix:
            continue
        key = ("prefix", prefix)
        if key in seen:
            continue
        seen.add(key)
        groups.append(
            {
                "group_name": f"top_{len(prefix)}",
                "feature_ids": list(prefix),
                "group_type": "prefix",
            }
        )
    return groups


def build_intervention_settings(
    feature_groups: Sequence[dict],
    *,
    amplify_factors: Sequence[float],
    observed_max_scales: Sequence[float],
) -> list[dict]:
    settings = []
    for group in feature_groups:
        for factor in amplify_factors:
            settings.append(
                {
                    "setting_name": f"{group['group_name']}__amplify_x{factor:g}",
                    "feature_ids": list(group["feature_ids"]),
                    "group_name": group["group_name"],
                    "group_type": group["group_type"],
                    "intervention_mode": "amplify",
                    "amplify_factor": float(factor),
                    "observed_max_scale": None,
                }
            )
        for scale in observed_max_scales:
            settings.append(
                {
                    "setting_name": f"{group['group_name']}__set_observed_max_x{scale:g}",
                    "feature_ids": list(group["feature_ids"]),
                    "group_name": group["group_name"],
                    "group_type": group["group_type"],
                    "intervention_mode": "set_observed_max",
                    "amplify_factor": None,
                    "observed_max_scale": float(scale),
                }
            )
    return settings


def build_manual_batch(dataset: WaymoE2E, dataset_indices: Sequence[int]) -> dict:
    return collate_with_images([dict(dataset[int(idx)]) for idx in dataset_indices])


@contextlib.contextmanager
def intervene_selected_features_hook(
    target_layer,
    sae: SparseAutoencoder,
    feature_ids: Sequence[int],
    mode: str,
    amplify_factor: float | None = None,
    target_values: torch.Tensor | None = None,
):
    feature_ids_t = torch.as_tensor(feature_ids, dtype=torch.long)

    def replace_with_modified_sae(module, inputs, output):
        del module, inputs
        flat_output = output.reshape(-1, output.size(-1))
        hidden, preprocess_stats = sae.encode(
            flat_output,
            return_preprocess_stats=True,
        )
        if feature_ids_t.numel() > 0:
            target_ids = feature_ids_t.to(device=hidden.device)
            hidden = hidden.clone()
            if mode == "amplify":
                if amplify_factor is None:
                    raise ValueError("Amplify intervention requires amplify_factor")
                hidden[:, target_ids] = hidden[:, target_ids] * amplify_factor
            elif mode == "set_observed_max":
                if target_values is None:
                    raise ValueError("set_observed_max intervention requires target_values")
                hidden[:, target_ids] = target_values.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                ).view(1, -1)
            else:
                raise ValueError(f"Unsupported intervention mode: {mode}")
        reconstructed = sae.decode_to_input(hidden, preprocess_stats=preprocess_stats)
        return reconstructed.view_as(output)

    handle = target_layer.register_forward_hook(replace_with_modified_sae)
    try:
        yield
    finally:
        handle.remove()


def run_patched_trajectory(
    *,
    model,
    sae: SparseAutoencoder,
    target_layer,
    past: torch.Tensor,
    intent: torch.Tensor,
    images,
    feature_ids: Sequence[int],
    mode: str,
    amplify_factor: float | None,
    target_values: torch.Tensor | None,
) -> torch.Tensor:
    with torch.no_grad():
        with intervene_selected_features_hook(
            target_layer=target_layer,
            sae=sae,
            feature_ids=feature_ids,
            mode=mode,
            amplify_factor=amplify_factor,
            target_values=target_values,
        ):
            output = model({"PAST": past, "IMAGES": images, "INTENT": intent})
    return select_best_trajectory(output).detach().cpu()


def summarize_setting(rows: Sequence[dict]) -> dict:
    if not rows:
        return {}
    endpoint_drop = torch.tensor([row["forward_endpoint_drop_m"] for row in rows], dtype=torch.float64)
    mean_drop = torch.tensor([row["forward_mean_drop_m"] for row in rows], dtype=torch.float64)
    traj_shift = torch.tensor([row["trajectory_shift_l2_mean"] for row in rows], dtype=torch.float64)
    endpoint_distance_reduction = torch.tensor(
        [row["endpoint_distance_reduction_m"] for row in rows], dtype=torch.float64
    )
    path_length_reduction = torch.tensor(
        [row["path_length_reduction_m"] for row in rows], dtype=torch.float64
    )
    sorted_rows = sorted(rows, key=lambda row: row["forward_endpoint_drop_m"], reverse=True)
    return {
        "setting_name": rows[0]["setting_name"],
        "group_name": rows[0]["group_name"],
        "group_type": rows[0]["group_type"],
        "feature_ids": rows[0]["feature_ids"],
        "intervention_mode": rows[0]["intervention_mode"],
        "amplify_factor": rows[0]["amplify_factor"],
        "observed_max_scale": rows[0]["observed_max_scale"],
        "n_examples": len(rows),
        "mean_forward_endpoint_drop_m": float(endpoint_drop.mean().item()),
        "max_forward_endpoint_drop_m": float(endpoint_drop.max().item()),
        "mean_forward_mean_drop_m": float(mean_drop.mean().item()),
        "max_forward_mean_drop_m": float(mean_drop.max().item()),
        "mean_trajectory_shift_l2": float(traj_shift.mean().item()),
        "max_trajectory_shift_l2": float(traj_shift.max().item()),
        "mean_endpoint_distance_reduction_m": float(endpoint_distance_reduction.mean().item()),
        "mean_path_length_reduction_m": float(path_length_reduction.mean().item()),
        "fraction_positive_endpoint_drop": float((endpoint_drop > 0).to(torch.float64).mean().item()),
        "fraction_positive_mean_drop": float((mean_drop > 0).to(torch.float64).mean().item()),
        "top_examples": sorted_rows[:10],
    }


def endpoint_distance(traj: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
    current_xy = past[:, -1, :2]
    return torch.norm(traj[:, -1] - current_xy, dim=-1)


def path_length(traj: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([past[:, -1:, :2], traj[:, :-1]], dim=1)
    return torch.norm(traj - prev, dim=-1).sum(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_json", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--sae_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_checkpoint_path",
        default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt",
        type=str,
    )
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--n_items", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=3)
    parser.add_argument("--camera_indices", type=str, default=None)
    parser.add_argument(
        "--feature_key",
        type=str,
        default="causal_single_feature_responsibility.top_single_feature_responsibility",
    )
    parser.add_argument("--feature_limit", type=int, default=16)
    parser.add_argument("--feature_counts", type=str, default="1,4")
    parser.add_argument("--include_single_feature_sweep", action="store_true")
    parser.add_argument("--amplify_factors", type=str, default="1000")
    parser.add_argument("--observed_max_scales", type=str, default="")
    parser.add_argument(
        "--frame_filter",
        type=str,
        default="clean",
        choices=["clean", "object_present", "all"],
        help="Which frames to evaluate: no stop sign, stop-sign present, or all",
    )
    parser.add_argument("--moving_speed_thresh", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    analysis_json_path = resolve_input_path(args.analysis_json, description="analysis JSON")
    detections_path = resolve_input_path(args.detections, description="detections artifact")
    sae_checkpoint_path = resolve_input_path(args.sae_checkpoint_path, description="SAE checkpoint")
    model_checkpoint_path = resolve_input_path(args.model_checkpoint_path, description="model checkpoint")
    data_dir = resolve_input_path(args.data_dir, description="dataset directory")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = load_json(analysis_json_path)
    split = args.split or analysis.get("split", "train")
    index_file = resolve_input_path(
        args.index_file or analysis.get("index_file", default_index_file(split)),
        description="dataset index file",
    )

    raw_camera_indices = parse_csv_ints(args.camera_indices)
    if raw_camera_indices:
        camera_indices = raw_camera_indices
    else:
        camera_indices = [analysis.get("front_camera_idx", 1)]

    feature_rows = choose_feature_rows(
        analysis,
        feature_key=args.feature_key,
        feature_limit=args.feature_limit,
    )
    feature_groups = choose_feature_groups(
        feature_rows,
        feature_counts=parse_csv_ints(args.feature_counts),
        include_single_feature_sweep=args.include_single_feature_sweep,
    )
    if not feature_groups:
        raise RuntimeError("No feature groups were selected")

    intervention_settings = build_intervention_settings(
        feature_groups,
        amplify_factors=parse_csv_floats(args.amplify_factors),
        observed_max_scales=parse_csv_floats(args.observed_max_scales),
    )
    if not intervention_settings:
        raise RuntimeError("No intervention settings were selected")

    categories, frame_to_record = load_detection_artifacts(str(detections_path))
    object_label_id, object_label_name = resolve_object_label(
        categories,
        analysis.get("object_label_name", "stop sign"),
        analysis.get("object_label_id"),
    )
    score_thresh = float(analysis.get("score_thresh", 0.4))

    model, sae, _ = prepare_model_and_sae(
        model_checkpoint_path=str(model_checkpoint_path),
        sae_checkpoint_path=str(sae_checkpoint_path),
        block_idx=args.block_idx,
        device=device,
    )
    target_layer = get_sae_target_layer(model, args.block_idx)

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

    accepted_rows: list[dict] = []
    raw_frames_scanned = 0
    missing_detection_frames = 0
    filtered_out_by_object = 0
    skipped_low_speed = 0
    running_dataset_idx = 0

    for batch in tqdm(loader, desc="Selecting evaluation frames"):
        if len(accepted_rows) >= args.max_examples:
            break

        frame_names = batch["NAME"]
        batch_size = len(frame_names)
        raw_frames_scanned += batch_size
        past_cpu = batch["PAST"]

        for sample_idx, frame_name in enumerate(frame_names):
            if len(accepted_rows) >= args.max_examples:
                break

            record = frame_to_record.get(frame_name)
            if record is None:
                missing_detection_frames += 1
                continue

            has_object = frame_has_object(
                record=record,
                label_id=object_label_id,
                score_thresh=score_thresh,
                camera_indices=camera_indices,
            )
            if args.frame_filter == "clean" and has_object:
                filtered_out_by_object += 1
                continue
            if args.frame_filter == "object_present" and not has_object:
                filtered_out_by_object += 1
                continue

            current_speed = past_current_speed(past_cpu[sample_idx])
            if current_speed < args.moving_speed_thresh:
                skipped_low_speed += 1
                continue

            accepted_rows.append(
                {
                    "dataset_idx": running_dataset_idx + sample_idx,
                    "frame_name": frame_name,
                    "has_object": bool(has_object),
                    "current_speed": float(current_speed),
                }
            )

        running_dataset_idx += batch_size

    if not accepted_rows:
        raise RuntimeError("No eligible frames matched the requested filter")

    candidate_feature_ids = sorted(
        {
            feature_id
            for setting in intervention_settings
            for feature_id in setting["feature_ids"]
        }
    )
    feature_to_col = {feature_id: idx for idx, feature_id in enumerate(candidate_feature_ids)}
    observed_max = torch.full((len(candidate_feature_ids),), float("-inf"), dtype=torch.float64)

    for start in tqdm(range(0, len(accepted_rows), args.batch_size), desc="Collecting baseline activation stats"):
        batch_rows = accepted_rows[start : start + args.batch_size]
        batch = build_manual_batch(dataset, [row["dataset_idx"] for row in batch_rows])
        past = batch["PAST"].to(device)
        intent = batch["INTENT"].to(device)
        images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
        baseline_hidden, _ = run_model_hidden_and_trajectory(
            model=model,
            sae=sae,
            past=past,
            intent=intent,
            images=images,
        )
        hidden_subset = baseline_hidden[:, candidate_feature_ids]
        observed_max = torch.maximum(observed_max, hidden_subset.max(dim=0).values)

    setting_rows: list[dict] = []
    combined_rows: list[dict] = []

    for setting in tqdm(intervention_settings, desc="Evaluating interventions"):
        rows_for_setting: list[dict] = []
        target_values = None
        if setting["intervention_mode"] == "set_observed_max":
            scale = float(setting["observed_max_scale"])
            target_values = torch.tensor(
                [
                    float(observed_max[feature_to_col[feature_id]].item()) * scale
                    for feature_id in setting["feature_ids"]
                ],
                dtype=torch.float32,
            )

        for start in range(0, len(accepted_rows), args.batch_size):
            batch_rows = accepted_rows[start : start + args.batch_size]
            batch = build_manual_batch(dataset, [row["dataset_idx"] for row in batch_rows])
            past = batch["PAST"].to(device)
            past_cpu = batch["PAST"].detach().cpu()
            intent = batch["INTENT"].to(device)
            images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)

            baseline_hidden, baseline_traj = run_model_hidden_and_trajectory(
                model=model,
                sae=sae,
                past=past,
                intent=intent,
                images=images,
            )
            del baseline_hidden

            patched_traj = run_patched_trajectory(
                model=model,
                sae=sae,
                target_layer=target_layer,
                past=past,
                intent=intent,
                images=images,
                feature_ids=setting["feature_ids"],
                mode=setting["intervention_mode"],
                amplify_factor=setting["amplify_factor"],
                target_values=target_values,
            )

            baseline_traj = baseline_traj.to(torch.float64)
            patched_traj = patched_traj.to(torch.float64)
            forward_endpoint_drop = baseline_traj[:, -1, 0] - patched_traj[:, -1, 0]
            forward_mean_drop = baseline_traj[..., 0].mean(dim=-1) - patched_traj[..., 0].mean(dim=-1)
            traj_shift = torch.norm(baseline_traj - patched_traj, dim=-1).mean(dim=-1)
            baseline_endpoint_distance = endpoint_distance(baseline_traj, past_cpu)
            patched_endpoint_distance = endpoint_distance(patched_traj, past_cpu)
            baseline_path_length = path_length(baseline_traj, past_cpu)
            patched_path_length = path_length(patched_traj, past_cpu)

            for local_idx, row in enumerate(batch_rows):
                result_row = {
                    "setting_name": setting["setting_name"],
                    "group_name": setting["group_name"],
                    "group_type": setting["group_type"],
                    "feature_ids": list(setting["feature_ids"]),
                    "intervention_mode": setting["intervention_mode"],
                    "amplify_factor": setting["amplify_factor"],
                    "observed_max_scale": setting["observed_max_scale"],
                    "dataset_idx": int(row["dataset_idx"]),
                    "frame_name": row["frame_name"],
                    "has_object": row["has_object"],
                    "current_speed": row["current_speed"],
                    "forward_endpoint_drop_m": float(forward_endpoint_drop[local_idx].item()),
                    "forward_mean_drop_m": float(forward_mean_drop[local_idx].item()),
                    "trajectory_shift_l2_mean": float(traj_shift[local_idx].item()),
                    "baseline_endpoint_distance_m": float(baseline_endpoint_distance[local_idx].item()),
                    "patched_endpoint_distance_m": float(patched_endpoint_distance[local_idx].item()),
                    "endpoint_distance_reduction_m": float(
                        (baseline_endpoint_distance[local_idx] - patched_endpoint_distance[local_idx]).item()
                    ),
                    "baseline_path_length_m": float(baseline_path_length[local_idx].item()),
                    "patched_path_length_m": float(patched_path_length[local_idx].item()),
                    "path_length_reduction_m": float(
                        (baseline_path_length[local_idx] - patched_path_length[local_idx]).item()
                    ),
                }
                rows_for_setting.append(result_row)
                combined_rows.append(result_row)

        setting_rows.append(summarize_setting(rows_for_setting))

    setting_rows = [row for row in setting_rows if row]
    setting_rows.sort(
        key=lambda row: (
            row["mean_forward_endpoint_drop_m"],
            row["max_forward_endpoint_drop_m"],
            row["mean_trajectory_shift_l2"],
        ),
        reverse=True,
    )

    summary = {
        "analysis_json": str(analysis_json_path),
        "detections": str(detections_path),
        "sae_checkpoint_path": str(sae_checkpoint_path),
        "model_checkpoint_path": str(model_checkpoint_path),
        "data_dir": str(data_dir),
        "index_file": str(index_file),
        "split": split,
        "object_label_name": object_label_name,
        "object_label_id": object_label_id,
        "score_thresh": score_thresh,
        "frame_filter": args.frame_filter,
        "camera_indices": camera_indices,
        "moving_speed_thresh": args.moving_speed_thresh,
        "feature_key": args.feature_key,
        "feature_limit": args.feature_limit,
        "feature_counts": parse_csv_ints(args.feature_counts),
        "include_single_feature_sweep": args.include_single_feature_sweep,
        "amplify_factors": parse_csv_floats(args.amplify_factors),
        "observed_max_scales": parse_csv_floats(args.observed_max_scales),
        "raw_frames_scanned": raw_frames_scanned,
        "eligible_frames_evaluated": len(accepted_rows),
        "missing_detection_frames": missing_detection_frames,
        "filtered_out_by_object": filtered_out_by_object,
        "skipped_low_speed": skipped_low_speed,
        "feature_rows_used": feature_rows,
        "observed_feature_maxima": {
            str(feature_id): float(observed_max[feature_to_col[feature_id]].item())
            for feature_id in candidate_feature_ids
        },
        "settings_ranked_by_mean_forward_endpoint_drop": setting_rows,
        "best_setting": setting_rows[0] if setting_rows else None,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(output_dir / "setting_summary.csv", setting_rows)
    write_csv(output_dir / "all_setting_results.csv", combined_rows)

    if setting_rows:
        best_name = setting_rows[0]["setting_name"]
        best_rows = [row for row in combined_rows if row["setting_name"] == best_name]
        best_rows.sort(key=lambda row: row["forward_endpoint_drop_m"], reverse=True)
        write_csv(output_dir / "best_setting_top_examples.csv", best_rows[:50])

    print(
        f"Evaluated {len(intervention_settings)} intervention settings over "
        f"{len(accepted_rows)} frames. Best mean forward endpoint drop="
        f"{setting_rows[0]['mean_forward_endpoint_drop_m']:.6f} m"
        if setting_rows
        else "No intervention settings were evaluated"
    )
    print(f"Saved braking-intervention sweep to {output_dir}")


if __name__ == "__main__":
    main()
