"""
Visualize trajectory changes when target-object detections are masked from frames.
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from analyze_sae_masked_object_neurons import (
    apply_detection_masks,
    collect_mask_boxes,
    filter_detection_matches,
    parse_csv_ints,
    prepare_model_and_sae,
    run_model_hidden_and_trajectory,
)
from analyze_sae_object_neurons import default_index_file, load_detection_artifacts
from loader import WaymoE2E, collate_with_images
from view_sae_object_analysis import (
    CAMERA_SLOT_TO_PROTO_NAME,
    load_front_camera_projection_assets,
    overlay_projected_trajectory,
    projected_visibility_score,
)


def load_analysis_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_single_batch(sample: dict) -> dict:
    return collate_with_images([dict(sample)])


def slot_to_camera_name(camera_slot: int) -> int:
    if camera_slot not in CAMERA_SLOT_TO_PROTO_NAME:
        raise ValueError(
            f"Camera slot {camera_slot} is not supported for projection. "
            f"Expected one of {sorted(CAMERA_SLOT_TO_PROTO_NAME)}."
        )
    return CAMERA_SLOT_TO_PROTO_NAME[camera_slot]


def feature_ids_from_analysis(analysis: dict, key: str, feature_count: int) -> List[int]:
    rows = analysis.get(key, [])
    return [int(row["feature_idx"]) for row in rows[:feature_count]]


def mask_boxes_on_numpy_image(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    fill_value: int = 0,
    margin: int = 0,
) -> np.ndarray:
    masked = image.copy()
    height, width = masked.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box
        left = max(0, min(width, int(np.floor(float(x1))) - margin))
        top = max(0, min(height, int(np.floor(float(y1))) - margin))
        right = max(0, min(width, int(np.ceil(float(x2))) + margin))
        bottom = max(0, min(height, int(np.ceil(float(y2))) + margin))
        if right <= left or bottom <= top:
            continue
        masked[top:bottom, left:right, :] = fill_value
    return masked


def filter_detection_record(
    detection_record: dict,
    target_label_ids: Sequence[int],
    score_thresh: float,
) -> dict:
    boxes, labels, label_names, scores = filter_detection_matches(
        detection_record=detection_record,
        target_label_ids=target_label_ids,
        score_thresh=score_thresh,
    )
    return {
        "boxes": boxes,
        "labels": labels,
        "label_names": label_names,
        "scores": scores,
    }


def draw_detection_boxes(ax, detection_record: dict, color: str = "gold"):
    for box, label_name, score in zip(
        detection_record.get("boxes", []),
        detection_record.get("label_names", []),
        detection_record.get("scores", []),
    ):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 5),
            f"{label_name}: {score:.2f}",
            color=color,
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.4, pad=1),
        )


def save_masked_example_figure(
    output_path: Path,
    frame_name: str,
    masked_label_names: Sequence[str],
    raw_image: np.ndarray,
    masked_image: np.ndarray,
    baseline_overlay: np.ndarray,
    masked_overlay: np.ndarray,
    detection_record: dict,
    trajectory_shift: float,
    reduced_score: float,
    increased_score: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    axes[0].imshow(raw_image)
    draw_detection_boxes(axes[0], detection_record)
    axes[0].set_title(f"Original image\n{frame_name}")
    axes[0].axis("off")

    axes[1].imshow(masked_image)
    axes[1].set_title("Masked image")
    axes[1].axis("off")

    axes[2].imshow(baseline_overlay)
    axes[2].set_title("Baseline projected trajectory")
    axes[2].axis("off")

    axes[3].imshow(masked_overlay)
    axes[3].set_title("Masked-input projected trajectory")
    axes[3].axis("off")

    label_text = ", ".join(masked_label_names) if masked_label_names else "target labels"
    fig.suptitle(
        f"{label_text} | traj_shift={trajectory_shift:.4f} | "
        f"reduced_score={reduced_score:.4f} | increased_score={increased_score:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_json", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--sae_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_checkpoint_path", default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt", type=str)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--n_items", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=3)
    parser.add_argument(
        "--camera_indices",
        type=str,
        default=None,
        help="Optional comma-separated image slots to mask. Defaults to the analysis JSON camera_indices.",
    )
    parser.add_argument("--feature_count", type=int, default=8)
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--min_visibility", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    analysis = load_analysis_json(args.analysis_json)
    score_thresh = float(analysis.get("score_thresh", 0.4))
    object_label_ids = [int(label_id) for label_id in analysis.get("object_label_ids", [])]
    object_label_names = analysis.get("object_label_names", [])
    if not object_label_ids:
        raise ValueError("Analysis JSON is missing object_label_ids")

    camera_indices = parse_csv_ints(args.camera_indices)
    if not camera_indices:
        camera_indices = analysis.get("camera_indices", [1]) or [1]
    display_camera_slot = int(camera_indices[0])
    display_camera_name = slot_to_camera_name(display_camera_slot)

    reduced_feature_ids = feature_ids_from_analysis(
        analysis,
        "top_reduced_by_masking",
        args.feature_count,
    )
    increased_feature_ids = feature_ids_from_analysis(
        analysis,
        "top_increased_by_masking",
        args.feature_count,
    )

    _, frame_to_record = load_detection_artifacts(args.detections)
    model, sae, _ = prepare_model_and_sae(
        model_checkpoint_path=args.model_checkpoint_path,
        sae_checkpoint_path=args.sae_checkpoint_path,
        block_idx=args.block_idx,
        device=device,
    )

    index_file = args.index_file or default_index_file(analysis.get("split", "val"))
    dataset = WaymoE2E(indexFile=index_file, data_dir=args.data_dir, n_items=None)
    if args.n_items is not None:
        dataset.indexes = dataset.indexes[: args.n_items]

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    candidates = []
    running_dataset_idx = 0

    with torch.no_grad():
        for batch in loader:
            frame_names = batch["NAME"]
            batch_has_maskable_frame = False
            for frame_name in frame_names:
                _, _, total_boxes = collect_mask_boxes(
                    record=frame_to_record.get(frame_name),
                    target_label_ids=object_label_ids,
                    score_thresh=score_thresh,
                    camera_indices=camera_indices,
                )
                if total_boxes > 0:
                    batch_has_maskable_frame = True
                    break
            if not batch_has_maskable_frame:
                running_dataset_idx += len(frame_names)
                continue

            past = batch["PAST"].to(device)
            intent = batch["INTENT"].to(device)
            images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
            masked_images, masked_box_counts, masked_label_names = apply_detection_masks(
                decoded_images=images,
                frame_names=frame_names,
                frame_to_record=frame_to_record,
                target_label_ids=object_label_ids,
                score_thresh=score_thresh,
                camera_indices=camera_indices,
                fill_value=float(analysis.get("mask_fill_value", 0.0)),
                margin=int(analysis.get("mask_margin", 0)),
            )

            if not any(masked_box_counts):
                running_dataset_idx += len(frame_names)
                continue

            baseline_hidden, baseline_traj = run_model_hidden_and_trajectory(
                model=model,
                sae=sae,
                past=past,
                intent=intent,
                images=images,
            )
            masked_hidden, masked_traj = run_model_hidden_and_trajectory(
                model=model,
                sae=sae,
                past=past,
                intent=intent,
                images=masked_images,
            )

            hidden_delta = baseline_hidden - masked_hidden
            traj_shift = torch.norm(baseline_traj - masked_traj, dim=-1).mean(dim=-1)

            for sample_idx, frame_name in enumerate(frame_names):
                if masked_box_counts[sample_idx] == 0:
                    continue

                reduced_score = (
                    float(hidden_delta[sample_idx, reduced_feature_ids].mean().item())
                    if reduced_feature_ids
                    else 0.0
                )
                increased_score = (
                    float((-hidden_delta[sample_idx, increased_feature_ids]).mean().item())
                    if increased_feature_ids
                    else 0.0
                )
                candidates.append(
                    {
                        "frame_name": frame_name,
                        "dataset_idx": running_dataset_idx + sample_idx,
                        "masked_box_count": int(masked_box_counts[sample_idx]),
                        "masked_label_names": masked_label_names[sample_idx],
                        "trajectory_shift_l2_mean": float(traj_shift[sample_idx].item()),
                        "reduced_score": reduced_score,
                        "increased_score": increased_score,
                    }
                )

            running_dataset_idx += len(frame_names)

    candidates.sort(key=lambda row: row["trajectory_shift_l2_mean"], reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_examples = []
    for candidate in candidates:
        if len(selected_examples) >= args.num_examples:
            break

        sample = dataset[candidate["dataset_idx"]]
        batch = build_single_batch(sample)
        record = frame_to_record.get(sample["NAME"])
        if record is None:
            continue

        past = batch["PAST"].to(device)
        future = batch["FUTURE"].detach().cpu()
        intent = batch["INTENT"].to(device)
        images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
        masked_images, masked_box_counts, masked_label_names = apply_detection_masks(
            decoded_images=images,
            frame_names=batch["NAME"],
            frame_to_record=frame_to_record,
            target_label_ids=object_label_ids,
            score_thresh=score_thresh,
            camera_indices=camera_indices,
            fill_value=float(analysis.get("mask_fill_value", 0.0)),
            margin=int(analysis.get("mask_margin", 0)),
        )
        if not masked_box_counts[0]:
            continue

        baseline_hidden, baseline_traj = run_model_hidden_and_trajectory(
            model=model,
            sae=sae,
            past=past,
            intent=intent,
            images=images,
        )
        masked_hidden, masked_traj = run_model_hidden_and_trajectory(
            model=model,
            sae=sae,
            past=past,
            intent=intent,
            images=masked_images,
        )
        hidden_delta = baseline_hidden - masked_hidden
        reduced_score = (
            float(hidden_delta[0, reduced_feature_ids].mean().item()) if reduced_feature_ids else 0.0
        )
        increased_score = (
            float((-hidden_delta[0, increased_feature_ids]).mean().item()) if increased_feature_ids else 0.0
        )
        trajectory_shift = float(torch.norm(baseline_traj - masked_traj, dim=-1).mean(dim=-1)[0].item())

        raw_image, calibration = load_front_camera_projection_assets(
            dataset,
            candidate["dataset_idx"],
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
                trajectory_xy=masked_traj[0],
                future_xy=future[0],
            ),
        )
        if visibility < args.min_visibility:
            continue

        detection_record = filter_detection_record(
            detection_record=record["detections"].get(
                str(display_camera_slot),
                {"boxes": [], "labels": [], "label_names": [], "scores": []},
            ),
            target_label_ids=object_label_ids,
            score_thresh=score_thresh,
        )
        masked_image = mask_boxes_on_numpy_image(
            raw_image,
            boxes=detection_record["boxes"],
            fill_value=int(analysis.get("mask_fill_value", 0)),
            margin=int(analysis.get("mask_margin", 0)),
        )
        baseline_overlay = overlay_projected_trajectory(
            image=raw_image,
            calibration=calibration,
            trajectory_xy=baseline_traj[0],
            future_xy=future[0],
            pred_color=(255, 0, 0),
        )
        masked_overlay = overlay_projected_trajectory(
            image=masked_image,
            calibration=calibration,
            trajectory_xy=masked_traj[0],
            future_xy=future[0],
            pred_color=(255, 165, 0),
        )

        example_idx = len(selected_examples) + 1
        output_path = output_dir / f"masked_example_{example_idx:02d}.png"
        save_masked_example_figure(
            output_path=output_path,
            frame_name=sample["NAME"],
            masked_label_names=masked_label_names[0],
            raw_image=raw_image,
            masked_image=masked_image,
            baseline_overlay=baseline_overlay,
            masked_overlay=masked_overlay,
            detection_record=detection_record,
            trajectory_shift=trajectory_shift,
            reduced_score=reduced_score,
            increased_score=increased_score,
        )
        selected_examples.append(
            {
                "frame_name": sample["NAME"],
                "dataset_idx": candidate["dataset_idx"],
                "masked_box_count": int(masked_box_counts[0]),
                "masked_label_names": masked_label_names[0],
                "trajectory_shift_l2_mean": trajectory_shift,
                "reduced_score": reduced_score,
                "increased_score": increased_score,
                "visibility_score": int(visibility),
                "figure_path": str(output_path),
            }
        )

    summary = {
        "object_label_names": object_label_names,
        "object_label_ids": object_label_ids,
        "score_thresh": score_thresh,
        "camera_indices": camera_indices,
        "reduced_feature_ids": reduced_feature_ids,
        "increased_feature_ids": increased_feature_ids,
        "num_examples_requested": args.num_examples,
        "num_examples_saved": len(selected_examples),
        "selected_examples": selected_examples,
    }
    (output_dir / "viewer_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"Saved {len(selected_examples)} masked viewer example(s) for "
        f"{', '.join(object_label_names)} to {output_dir}"
    )


if __name__ == "__main__":
    main()
