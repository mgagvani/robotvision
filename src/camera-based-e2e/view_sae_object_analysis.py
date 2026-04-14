"""
Visualize object-associated SAE feature interventions on camera frames.
"""

import argparse
import contextlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.multiprocessing as mp

from analyze_sae_object_neurons import (
    compute_hidden_activations,
    default_index_file,
    frame_has_object,
    get_sae_state_dict,
    load_detection_artifacts,
    resolve_object_label,
)
from loader import WaymoE2E, collate_with_images
from models.base_model import LitModel
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel
from protos import e2e_pb2
from sparseAE import SparseAE
from viz_camera_projection import (
    CAMERA_FRONT,
    CAMERA_FRONT_LEFT,
    CAMERA_FRONT_RIGHT,
    decode_image,
    draw_trajectory_on_image,
    get_camera_calibration,
    project_trajectory_to_image,
)

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


# Dataset image slots follow the exported/default order:
# [FRONT_LEFT, FRONT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT, REAR_RIGHT, REAR, REAR_LEFT]
CAMERA_SLOT_TO_PROTO_NAME = {
    0: CAMERA_FRONT_LEFT,
    1: CAMERA_FRONT,
    2: CAMERA_FRONT_RIGHT,
}


def select_best_trajectory(output: dict) -> torch.Tensor:
    pred = output["trajectory"]
    scores = output.get("scores", None)
    bsz = pred.size(0)
    pred = pred.view(bsz, -1, pred.size(-2), 2)
    if scores is not None and pred.size(1) > 1:
        best_idx = scores.argmin(dim=1)
    else:
        best_idx = torch.zeros(bsz, dtype=torch.long, device=pred.device)
    return pred[torch.arange(bsz, device=pred.device), best_idx]


def prepare_model_and_sae(model_checkpoint_path: str, sae_checkpoint_path: str, block_idx: int, device: torch.device):
    submodel = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=40,
        n_blocks=4,
    )
    model = LitModel.load_from_checkpoint(model_checkpoint_path, model=submodel)
    model = model.to(device)
    model.eval()

    target_layer = model.model.blocks[block_idx].mlp[2]
    sae_checkpoint = torch.load(sae_checkpoint_path, map_location="cpu")
    sae_state = get_sae_state_dict(sae_checkpoint)
    encoder_weight = sae_state["encoder.weight"]
    dict_size, input_dim = encoder_weight.shape
    sae = SparseAE.build_from_state_dict(
        sae_state,
        target_model=model,
        input_dim=input_dim,
        dict_size=dict_size,
        compile_sae=False,
    )
    sae = sae.to(device)
    sae.eval()
    target_layer.register_forward_hook(sae.hook_fn)
    return model, sae, dict_size, input_dim, target_layer


def load_analysis_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_single_batch(sample: dict) -> dict:
    batch = collate_with_images([dict(sample)])
    return batch


def run_hidden_for_batch(model: LitModel, sae: SparseAE, batch: dict, device: torch.device) -> torch.Tensor:
    past = batch["PAST"].to(device)
    intent = batch["INTENT"].to(device)
    images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
    _ = model({"PAST": past, "IMAGES": images, "INTENT": intent})
    return compute_hidden_activations(sae, sae.internal_acts).detach().cpu().to(torch.float64)


@contextlib.contextmanager
def intervene_selected_features_hook(
    target_layer,
    sae: SparseAE,
    target_sample_idx: int,
    feature_ids: Sequence[int],
    mode: str,
    amplify_factor: float,
):
    feature_ids_t = torch.as_tensor(feature_ids, dtype=torch.long)

    def replace_with_modified_sae(module, inputs, output):
        x = output
        x_centered = x - sae.decoder.bias
        hidden = torch.relu(sae.encoder(x_centered))
        if feature_ids_t.numel() > 0:
            hidden = hidden.clone()
            source = hidden[target_sample_idx, :, feature_ids_t].clone()
            if mode == "zero":
                hidden[target_sample_idx, :, feature_ids_t] = 0.0
            elif mode == "amplify":
                hidden[target_sample_idx, :, feature_ids_t] = source * amplify_factor
            elif mode == "reverse":
                hidden[target_sample_idx, :, feature_ids_t] = source.flip(-1)
            else:
                raise ValueError(f"Unsupported intervention mode: {mode}")
        reconstructed = sae.decoder(hidden)
        return reconstructed

    handle = target_layer.register_forward_hook(replace_with_modified_sae)
    try:
        yield
    finally:
        handle.remove()


def run_model_outputs(
    model: LitModel,
    sae: SparseAE,
    target_layer,
    batch: dict,
    device: torch.device,
    feature_ids: Optional[Sequence[int]] = None,
    target_sample_idx: int = 0,
    intervention_mode: str = "zero",
    amplify_factor: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    past = batch["PAST"].to(device)
    future = batch["FUTURE"].to(device)
    intent = batch["INTENT"].to(device)
    images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
    model_inputs = {"PAST": past, "IMAGES": images, "INTENT": intent}

    with torch.no_grad():
        baseline = select_best_trajectory(model(model_inputs)).detach().cpu()
        if feature_ids is None or len(feature_ids) == 0:
            patched = baseline.clone()
        else:
            with intervene_selected_features_hook(
                target_layer,
                sae,
                target_sample_idx,
                feature_ids,
                mode=intervention_mode,
                amplify_factor=amplify_factor,
            ):
                patched = select_best_trajectory(model(model_inputs)).detach().cpu()

    return baseline, patched, [img.detach().cpu() for img in images], future.detach().cpu()


def draw_detection_boxes(ax, detection_record: dict, color: str = "lime"):
    for box, label_name, score in zip(
        detection_record["boxes"],
        detection_record["label_names"],
        detection_record["scores"],
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


def plot_trajectory_panel(ax, past: torch.Tensor, traj: torch.Tensor, future: torch.Tensor, title: str):
    past_xy = past[:, :2].cpu().numpy()
    traj_xy = traj.cpu().numpy()
    future_xy = future.cpu().numpy()

    ax.plot(past_xy[:, 0], past_xy[:, 1], "o-", color="tab:blue", label="past", linewidth=2, markersize=3)
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "o-", color="tab:red", label="pred", linewidth=2, markersize=3)
    ax.plot(future_xy[:, 0], future_xy[:, 1], "o-", color="tab:green", label="gt", linewidth=2, markersize=3)
    ax.scatter([past_xy[-1, 0]], [past_xy[-1, 1]], color="black", s=30)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")


def save_example_figure(
    output_path: Path,
    frame_name: str,
    object_name: str,
    raw_image: np.ndarray,
    baseline_overlay: np.ndarray,
    patched_overlay: np.ndarray,
    detection_record: dict,
    title_suffix: str,
    patched_title: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(raw_image)
    axes[0].set_title(f"{title_suffix} image\n{frame_name}")
    axes[0].axis("off")
    draw_detection_boxes(axes[0], detection_record)
    axes[0].text(
        10,
        20,
        object_name,
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.6),
    )
    axes[1].imshow(baseline_overlay)
    axes[1].set_title("Baseline projected trajectory")
    axes[1].axis("off")
    axes[2].imshow(patched_overlay)
    axes[2].set_title(patched_title)
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def intervention_title(mode: str) -> str:
    if mode == "zero":
        return "Zeroed feature trajectory"
    if mode == "amplify":
        return "Amplified feature trajectory"
    if mode == "reverse":
        return "Reversed-feature trajectory"
    return f"{mode} feature trajectory"


def parse_intervention_modes(spec: str) -> Tuple[str, str]:
    valid_modes = {"zero", "amplify", "reverse"}
    pieces = [piece.strip() for piece in spec.split(",") if piece.strip()]
    if len(pieces) == 1:
        pieces = [pieces[0], pieces[0]]
    if len(pieces) != 2:
        raise ValueError(
            "--intervention_modes must be a single mode or 'positive_mode,negative_mode'"
        )
    invalid = [piece for piece in pieces if piece not in valid_modes]
    if invalid:
        raise ValueError(f"Unsupported intervention mode(s): {', '.join(invalid)}")
    return pieces[0], pieces[1]


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


def load_front_camera_projection_assets(dataset: WaymoE2E, dataset_idx: int, camera_name: int = 1):
    filename, start_byte, byte_length = dataset.indexes[dataset_idx]
    file_path = Path(dataset.data_dir) / filename
    frame = e2e_pb2.E2EDFrame()
    with open(file_path, "rb") as f:
        f.seek(start_byte)
        frame.ParseFromString(f.read(byte_length))

    front_image = None
    for img in frame.frame.images:
        if img.name == camera_name:
            front_image = decode_image(img.image)
            break

    if front_image is None:
        raise RuntimeError(f"Camera {camera_name} not found for dataset index {dataset_idx}")

    intrinsic, extrinsic, dist_coeffs, width, height = get_camera_calibration(
        frame.frame.context.camera_calibrations,
        camera_name,
    )
    calibration = {
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "dist_coeffs": dist_coeffs,
        "width": width,
        "height": height,
    }
    return front_image, calibration


def overlay_projected_trajectory(
    image: np.ndarray,
    calibration: dict,
    trajectory_xy: torch.Tensor,
    future_xy: torch.Tensor,
    pred_color: Tuple[int, int, int],
) -> np.ndarray:
    projected_pred = project_trajectory_to_image(
        trajectory_xy.cpu().numpy(),
        calibration["intrinsic"],
        calibration["extrinsic"],
        calibration["dist_coeffs"],
        calibration["width"],
        calibration["height"],
    )
    projected_future = project_trajectory_to_image(
        future_xy.cpu().numpy(),
        calibration["intrinsic"],
        calibration["extrinsic"],
        calibration["dist_coeffs"],
        calibration["width"],
        calibration["height"],
    )

    out = draw_trajectory_on_image(
        image,
        projected_pred,
        pred_color,
        thickness=4,
        point_radius=5,
        alpha=0.9,
    )
    out = draw_trajectory_on_image(
        out,
        projected_future,
        (0, 255, 0),
        thickness=4,
        point_radius=5,
        alpha=0.9,
    )
    return out


def projected_visibility_score(
    calibration: dict,
    trajectory_xy: torch.Tensor,
    future_xy: torch.Tensor,
) -> int:
    projected_pred = project_trajectory_to_image(
        trajectory_xy.cpu().numpy(),
        calibration["intrinsic"],
        calibration["extrinsic"],
        calibration["dist_coeffs"],
        calibration["width"],
        calibration["height"],
    )
    projected_future = project_trajectory_to_image(
        future_xy.cpu().numpy(),
        calibration["intrinsic"],
        calibration["extrinsic"],
        calibration["dist_coeffs"],
        calibration["width"],
        calibration["height"],
    )
    pred_visible = int((~np.isnan(projected_pred).any(axis=1)).sum())
    future_visible = int((~np.isnan(projected_future).any(axis=1)).sum())
    return pred_visible + future_visible


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=3)
    parser.add_argument(
        "--camera_indices",
        type=str,
        default="1",
        help="Comma-separated image slots to score/display (0=FRONT_LEFT, 1=FRONT, 2=FRONT_RIGHT)",
    )
    parser.add_argument("--feature_count", type=int, default=8)
    parser.add_argument(
        "--intervention_modes",
        type=str,
        default="zero,amplify",
        help="Single mode for both examples or 'positive_mode,negative_mode'",
    )
    parser.add_argument("--amplify_factor", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    camera_indices = [int(piece.strip()) for piece in args.camera_indices.split(",") if piece.strip()]
    display_camera_slot = camera_indices[0] if camera_indices else 1
    display_camera_name = slot_to_camera_name(display_camera_slot)

    analysis = load_analysis_json(args.analysis_json)
    split = analysis.get("split", "val")
    score_thresh = float(analysis.get("score_thresh", 0.4))
    positive_mode, negative_mode = parse_intervention_modes(args.intervention_modes)
    categories, frame_to_record = load_detection_artifacts(args.detections)
    label_id, label_name = resolve_object_label(
        categories,
        analysis.get("object_label_name"),
        analysis.get("object_label_id"),
    )
    positive_feature_ids = feature_ids_from_analysis(
        analysis,
        "top_positive_association",
        args.feature_count,
    )
    negative_feature_ids = feature_ids_from_analysis(
        analysis,
        "top_negative_association",
        args.feature_count,
    )

    model, sae, dict_size, _, target_layer = prepare_model_and_sae(
        model_checkpoint_path=args.model_checkpoint_path,
        sae_checkpoint_path=args.sae_checkpoint_path,
        block_idx=args.block_idx,
        device=device,
    )

    index_file = args.index_file or default_index_file(split)
    dataset = WaymoE2E(indexFile=index_file, data_dir=args.data_dir, n_items=None)
    if args.n_items is not None:
        dataset.indexes = dataset.indexes[: args.n_items]

    positive_candidates = []
    negative_candidates = []

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    running_dataset_idx = 0
    with torch.no_grad():
        for batch in loader:
            hidden = run_hidden_for_batch(model, sae, batch, device)
            frame_names = batch["NAME"]
            labels = []
            for frame_name in frame_names:
                record = frame_to_record.get(frame_name)
                if record is None:
                    labels.append(0.0)
                else:
                    labels.append(float(frame_has_object(record, label_id, score_thresh, camera_indices=camera_indices)))

            y = torch.tensor(labels, dtype=torch.float64)
            pos_score = (
                hidden[:, positive_feature_ids].mean(dim=1)
                if positive_feature_ids
                else torch.zeros(hidden.size(0), dtype=torch.float64)
            )
            neg_score = (
                hidden[:, negative_feature_ids].mean(dim=1)
                if negative_feature_ids
                else torch.zeros(hidden.size(0), dtype=torch.float64)
            )
            for sample_idx, frame_name in enumerate(frame_names):
                dataset_idx = running_dataset_idx + sample_idx
                if y[sample_idx].item() > 0.5:
                    score = float(pos_score[sample_idx].item())
                    positive_candidates.append(
                        {"score": score, "dataset_idx": dataset_idx, "frame_name": frame_name}
                    )
                else:
                    score = float(neg_score[sample_idx].item())
                    negative_candidates.append(
                        {"score": score, "dataset_idx": dataset_idx, "frame_name": frame_name}
                    )
            running_dataset_idx += hidden.size(0)

    positive_candidates.sort(key=lambda row: row["score"], reverse=True)
    negative_candidates.sort(key=lambda row: row["score"], reverse=True)

    def choose_visible_example(candidates, feature_ids, intervention_mode):
        fallback = candidates[0] if candidates else {"score": float("-inf"), "dataset_idx": None, "frame_name": None}
        for example_info in candidates:
            sample = dataset[example_info["dataset_idx"]]
            batch = build_single_batch(sample)
            baseline_traj, _, _, future = run_model_outputs(
                model=model,
                sae=sae,
                target_layer=target_layer,
                batch=batch,
                device=device,
                feature_ids=feature_ids,
                target_sample_idx=0,
                intervention_mode=intervention_mode,
                amplify_factor=args.amplify_factor,
            )
            _, calibration = load_front_camera_projection_assets(
                dataset,
                example_info["dataset_idx"],
                camera_name=display_camera_name,
            )
            visibility = projected_visibility_score(
                calibration=calibration,
                trajectory_xy=baseline_traj[0],
                future_xy=future[0],
            )
            if visibility >= 2:
                selected = dict(example_info)
                selected["visibility_score"] = visibility
                return selected

        selected = dict(fallback)
        if selected["dataset_idx"] is not None:
            selected["visibility_score"] = 0
        return selected

    positive_best = choose_visible_example(positive_candidates, positive_feature_ids, positive_mode)
    negative_best = choose_visible_example(negative_candidates, negative_feature_ids, negative_mode)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Object: {label_name} (label_id={label_id})")
    print("Top positively associated SAE features from analysis JSON:")
    for row in analysis.get("top_positive_association", [])[: args.feature_count]:
        print(f"  feature {row['feature_idx']}: delta_mean={row['delta_mean']:.6f}")
    print("Top negatively associated SAE features from analysis JSON:")
    for row in analysis.get("top_negative_association", [])[: args.feature_count]:
        print(f"  feature {row['feature_idx']}: delta_mean={row['delta_mean']:.6f}")

    summary = {
        "object_label_name": label_name,
        "object_label_id": label_id,
        "split": split,
        "score_thresh": score_thresh,
        "positive_example": positive_best,
        "negative_example": negative_best,
        "positive_feature_ids": positive_feature_ids,
        "negative_feature_ids": negative_feature_ids,
        "positive_mode": positive_mode,
        "negative_mode": negative_mode,
        "amplify_factor": args.amplify_factor,
    }
    (output_dir / "correlation_summary.json").write_text(json.dumps(summary, indent=2))

    examples = [
        ("positive", positive_best, positive_feature_ids, positive_mode),
        ("negative", negative_best, negative_feature_ids, negative_mode),
    ]

    for tag, example_info, feature_ids, intervention_mode in examples:
        if example_info["dataset_idx"] is None:
            continue

        sample = dataset[example_info["dataset_idx"]]
        batch = build_single_batch(sample)
        record = frame_to_record[sample["NAME"]]
        baseline_traj, patched_traj, decoded_images, future = run_model_outputs(
            model=model,
            sae=sae,
            target_layer=target_layer,
            batch=batch,
            device=device,
            feature_ids=feature_ids,
            target_sample_idx=0,
            intervention_mode=intervention_mode,
            amplify_factor=args.amplify_factor,
        )

        raw_image, calibration = load_front_camera_projection_assets(
            dataset,
            example_info["dataset_idx"],
            camera_name=display_camera_name,
        )
        detection_record = record["detections"].get(
            str(display_camera_slot),
            {"boxes": [], "label_names": [], "scores": []},
        )
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
        save_example_figure(
            output_path=output_dir / f"{tag}_example.png",
            frame_name=sample["NAME"],
            object_name=label_name,
            raw_image=raw_image,
            baseline_overlay=baseline_overlay,
            patched_overlay=patched_overlay,
            detection_record=detection_record,
            title_suffix=tag.capitalize(),
            patched_title=intervention_title(intervention_mode),
        )

    print(f"Saved viewer outputs to {output_dir}")


if __name__ == "__main__":
    main()
