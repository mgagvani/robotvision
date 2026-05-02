"""
Visualize the strongest braking examples from an SAE intervention sweep.

Loads a sweep summary JSON, re-runs the best intervention (or a requested
setting), and saves figures that include:
1. Raw front-camera image.
2. Baseline projected trajectory on the image.
3. Patched projected trajectory on the image.
4. XY trajectory graph comparing past / baseline / patched / ground truth.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_sae_masked_object_neurons import prepare_model_and_sae, select_best_trajectory
from analyze_sae_object_neurons import default_index_file
from loader import WaymoE2E, collate_with_images
from models.sae import SparseAutoencoder
from new_sae_utils import get_sae_target_layer
from view_sae_object_analysis import (
    load_front_camera_projection_assets,
    overlay_projected_trajectory,
    plot_trajectory_panel,
    projected_visibility_score,
    slot_to_camera_name,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def resolve_input_path(raw_path: str | Path, *, description: str) -> Path:
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


def load_csv_rows(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_single_batch(sample: dict) -> dict:
    return collate_with_images([dict(sample)])


def intervention_title(mode: str, feature_ids: Sequence[int], amplify_factor: float | None) -> str:
    feature_text = ",".join(str(idx) for idx in feature_ids)
    if mode == "amplify":
        return f"Amplified features [{feature_text}] x{amplify_factor:g}"
    if mode == "set_observed_max":
        return f"Set features [{feature_text}] to observed max"
    return f"{mode} features [{feature_text}]"


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


def run_baseline_and_patched(
    *,
    model,
    sae: SparseAutoencoder,
    target_layer,
    batch: dict,
    device: torch.device,
    feature_ids: Sequence[int],
    mode: str,
    amplify_factor: float | None,
    target_values: torch.Tensor | None,
):
    past = batch["PAST"].to(device)
    intent = batch["INTENT"].to(device)
    images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
    model_inputs = {"PAST": past, "IMAGES": images, "INTENT": intent}

    with torch.no_grad():
        baseline_output = model(model_inputs)
        baseline_traj = select_best_trajectory(baseline_output).detach().cpu()
        with intervene_selected_features_hook(
            target_layer=target_layer,
            sae=sae,
            feature_ids=feature_ids,
            mode=mode,
            amplify_factor=amplify_factor,
            target_values=target_values,
        ):
            patched_output = model(model_inputs)
        patched_traj = select_best_trajectory(patched_output).detach().cpu()

    return baseline_traj, patched_traj, batch["FUTURE"].detach().cpu()


def save_example_figure(
    *,
    output_path: Path,
    frame_name: str,
    raw_image: np.ndarray,
    baseline_overlay: np.ndarray,
    patched_overlay: np.ndarray,
    past: torch.Tensor,
    baseline_traj: torch.Tensor,
    patched_traj: torch.Tensor,
    future: torch.Tensor,
    title_text: str,
    forward_endpoint_drop_m: float,
    trajectory_shift_l2_mean: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    axes[0].imshow(raw_image)
    axes[0].set_title(f"Raw front image\n{frame_name}")
    axes[0].axis("off")

    axes[1].imshow(baseline_overlay)
    axes[1].set_title("Baseline projected trajectory")
    axes[1].axis("off")

    axes[2].imshow(patched_overlay)
    axes[2].set_title(title_text)
    axes[2].axis("off")

    plot_trajectory_panel(
        axes[3],
        past[0],
        baseline_traj[0],
        future[0],
        title="XY trajectory graph",
    )
    patched_xy = patched_traj[0].cpu().numpy()
    axes[3].plot(
        patched_xy[:, 0],
        patched_xy[:, 1],
        "o-",
        color="tab:orange",
        linewidth=2,
        markersize=3,
        label="patched",
    )
    axes[3].legend(loc="best")

    fig.suptitle(
        f"forward_endpoint_drop={forward_endpoint_drop_m:.4f} m | "
        f"traj_shift_l2_mean={trajectory_shift_l2_mean:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_summary_bar_plot(output_path: Path, example_rows: Sequence[dict]) -> None:
    if not example_rows:
        return
    labels = [f"{idx+1}" for idx in range(len(example_rows))]
    endpoint_drops = [float(row["forward_endpoint_drop_m"]) for row in example_rows]
    traj_shifts = [float(row["trajectory_shift_l2_mean"]) for row in example_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(labels, endpoint_drops, color="tab:red")
    axes[0].set_ylabel("Forward endpoint drop (m)")
    axes[0].set_title("Top examples by braking effect")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, traj_shifts, color="tab:blue")
    axes[1].set_ylabel("Trajectory shift L2")
    axes[1].set_xlabel("Ranked example")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def choose_example_rows(
    *,
    summary_path: Path,
    selected_setting: dict,
    selection_mode: str,
    num_examples: int,
    random_seed: int,
) -> list[dict]:
    if selection_mode == "top":
        return list(selected_setting.get("top_examples", []))[:num_examples]

    results_csv_path = summary_path.parent / "all_setting_results.csv"
    if not results_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {results_csv_path} needed for random selection"
        )
    rows = load_csv_rows(results_csv_path)
    matching = [
        row for row in rows
        if row.get("setting_name") == selected_setting["setting_name"]
    ]
    if not matching:
        raise RuntimeError(
            f"No rows in {results_csv_path} matched setting {selected_setting['setting_name']}"
        )
    rng = random.Random(random_seed)
    if len(matching) <= num_examples:
        rng.shuffle(matching)
        return matching
    return rng.sample(matching, k=num_examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_summary_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--setting_name", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--selection_mode", type=str, default="top", choices=["top", "random"])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--min_visibility", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    summary_path = resolve_input_path(args.sweep_summary_json, description="sweep summary JSON")
    sweep = load_json(summary_path)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.setting_name is None:
        selected_setting = sweep["best_setting"]
    else:
        selected_setting = None
        for row in sweep.get("settings_ranked_by_mean_forward_endpoint_drop", []):
            if row.get("setting_name") == args.setting_name:
                selected_setting = row
                break
        if selected_setting is None:
            raise ValueError(f"Could not find setting_name={args.setting_name} in sweep summary")

    split = sweep.get("split", "train")
    index_file = resolve_input_path(
        sweep.get("index_file", default_index_file(split)),
        description="dataset index file",
    )
    data_dir = resolve_input_path(sweep["data_dir"], description="dataset directory")
    model_checkpoint_path = resolve_input_path(sweep["model_checkpoint_path"], description="model checkpoint")
    sae_checkpoint_path = resolve_input_path(sweep["sae_checkpoint_path"], description="SAE checkpoint")

    model, sae, _ = prepare_model_and_sae(
        model_checkpoint_path=str(model_checkpoint_path),
        sae_checkpoint_path=str(sae_checkpoint_path),
        block_idx=3,
        device=device,
    )
    target_layer = get_sae_target_layer(model, 3)
    dataset = WaymoE2E(indexFile=str(index_file), data_dir=str(data_dir), n_items=None)

    feature_ids = [int(x) for x in selected_setting["feature_ids"]]
    intervention_mode = str(selected_setting["intervention_mode"])
    amplify_factor = selected_setting.get("amplify_factor")
    if amplify_factor is not None:
        amplify_factor = float(amplify_factor)

    target_values = None
    if intervention_mode == "set_observed_max":
        maxima = sweep.get("observed_feature_maxima", {})
        target_values = torch.tensor(
            [float(maxima[str(feature_id)]) for feature_id in feature_ids],
            dtype=torch.float32,
        )
        scale = selected_setting.get("observed_max_scale")
        if scale is not None:
            target_values = target_values * float(scale)

    display_camera_slot = int((sweep.get("camera_indices") or [1])[0])
    display_camera_name = slot_to_camera_name(display_camera_slot)

    example_rows = choose_example_rows(
        summary_path=summary_path,
        selected_setting=selected_setting,
        selection_mode=args.selection_mode,
        num_examples=args.num_examples,
        random_seed=args.random_seed,
    )
    saved_rows = []
    for example_rank, row in enumerate(example_rows, start=1):
        dataset_idx = int(row["dataset_idx"])
        sample = dataset[dataset_idx]
        batch = build_single_batch(sample)

        baseline_traj, patched_traj, future = run_baseline_and_patched(
            model=model,
            sae=sae,
            target_layer=target_layer,
            batch=batch,
            device=device,
            feature_ids=feature_ids,
            mode=intervention_mode,
            amplify_factor=amplify_factor,
            target_values=target_values,
        )

        raw_image, calibration = load_front_camera_projection_assets(
            dataset,
            dataset_idx,
            camera_name=display_camera_name,
        )
        visibility = max(
            projected_visibility_score(
                calibration=calibration,
                trajectory_xy=baseline_traj[0],
                future_xy=future[0],
            ),
            projected_visibility_score(
                calibration=calibration,
                trajectory_xy=patched_traj[0],
                future_xy=future[0],
            ),
        )
        if visibility < args.min_visibility:
            continue

        baseline_overlay = overlay_projected_trajectory(
            image=raw_image,
            calibration=calibration,
            trajectory_xy=baseline_traj[0],
            future_xy=future[0],
            pred_color=(255, 0, 0),
        )
        patched_overlay = overlay_projected_trajectory(
            image=raw_image,
            calibration=calibration,
            trajectory_xy=patched_traj[0],
            future_xy=future[0],
            pred_color=(255, 165, 0),
        )

        figure_path = output_dir / f"example_{example_rank:02d}.png"
        save_example_figure(
            output_path=figure_path,
            frame_name=sample["NAME"],
            raw_image=raw_image,
            baseline_overlay=baseline_overlay,
            patched_overlay=patched_overlay,
            past=batch["PAST"],
            baseline_traj=baseline_traj,
            patched_traj=patched_traj,
            future=future,
            title_text=intervention_title(intervention_mode, feature_ids, amplify_factor),
            forward_endpoint_drop_m=float(row["forward_endpoint_drop_m"]),
            trajectory_shift_l2_mean=float(row["trajectory_shift_l2_mean"]),
        )
        saved_rows.append(
            {
                **row,
                "visibility_score": int(visibility),
                "figure_path": str(figure_path),
            }
        )

    save_summary_bar_plot(output_dir / "top_examples_summary.png", saved_rows)

    viewer_summary = {
        "selected_setting_name": selected_setting["setting_name"],
        "feature_ids": feature_ids,
        "intervention_mode": intervention_mode,
        "amplify_factor": amplify_factor,
        "observed_max_scale": selected_setting.get("observed_max_scale"),
        "selection_mode": args.selection_mode,
        "random_seed": args.random_seed,
        "num_examples_requested": args.num_examples,
        "num_examples_saved": len(saved_rows),
        "saved_examples": saved_rows,
    }
    (output_dir / "viewer_summary.json").write_text(json.dumps(viewer_summary, indent=2))

    print(f"Selected setting: {selected_setting['setting_name']}")
    print(f"Feature IDs: {feature_ids}")
    print(f"Saved {len(saved_rows)} visualization example(s) to {output_dir}")


if __name__ == "__main__":
    main()
