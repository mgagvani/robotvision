#!/usr/bin/env python3
"""Extract real selected trajectories for the Fig. 3 counterfactual panels.

This writes a tiny JSON cache for only the displayed paper examples. It does
not modify the large visual-gen analysis .pt files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
E2E_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(E2E_ROOT))

# Some cluster environments expose a Pillow package where PIL.__version__ is
# missing. torchmetrics imports during Lightning setup expect it at import time.
try:
    import PIL

    if not hasattr(PIL, "__version__"):
        PIL.__version__ = "0.0.0"
except Exception:
    pass

from analyze_sae_visual_gen_pt2 import jpeg_bytes_to_tensor  # noqa: E402
from sae_utils import planner_inputs_from_collated_batch as model_inputs_from_batch  # noqa: E402
from extract_planner_tok import load_model  # noqa: E402
from models.base_model import collate_with_images  # noqa: E402
from protos import e2e_pb2  # noqa: E402


DEFAULT_ROWS = [
    ("red_to_green", 742),
    ("add_stop_sign", 163),
    ("add_pedestrian", 173),
]

EXPECTED_FINAL_X_SIGN = {
    "red_to_green": 1,
    "green_to_red": -1,
    "add_stop_sign": -1,
    "remove_stop_sign": 1,
    "add_pedestrian": -1,
    "remove_pedestrian": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/work/hdd/bgxf/mgagvani/visual_gen_1000/manifest.jsonl"),
    )
    parser.add_argument(
        "--pair_csv",
        type=Path,
        default=Path(
            "/work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis/"
            "sae_visual_gen_pairs_block_3.csv"
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(
            "/work/nvme/bgxf/mgagvani/wod/waymo_end_to_end_camera_v1_0_0/"
            "waymo_open_dataset_end_to_end_camera_v_1_0_0"
        ),
    )
    parser.add_argument("--index_file", type=Path, default=E2E_ROOT / "index_val.pkl")
    parser.add_argument(
        "--token_path",
        type=Path,
        default=E2E_ROOT / "sae_run_root" / "tokens" / "planner_tokens_val.pt",
    )
    parser.add_argument(
        "--planner_checkpoint",
        type=Path,
        default=E2E_ROOT / "sae_run_root" / "camera-e2e-epoch=04-val_loss=2.90.ckpt",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional direction=dataset_idx overrides.",
    )
    parser.add_argument(
        "--directions",
        nargs="*",
        default=None,
        help="Auto-select examples for these edit directions.",
    )
    parser.add_argument("--per_direction", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--expected_forward_delta",
        action="store_true",
        help="Keep only samples whose selected edited trajectory changes final x in the expected direction.",
    )
    parser.add_argument("--candidate_pool", type=int, default=40)
    parser.add_argument(
        "--exclude_cache",
        nargs="*",
        type=Path,
        default=None,
        help="JSON trajectory caches whose dataset_idx values should be skipped.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=E2E_ROOT
        / "visualizations"
        / "paper"
        / "fig3_counterfactual_trajectories.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def excluded_dataset_idxs(paths: list[Path] | None) -> set[int]:
    excluded = set()
    for path in paths or []:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("samples", []):
            excluded.add(int(row["dataset_idx"]))
    return excluded


def sample_plan(
    items: list[str] | None,
    *,
    pair_rows: list[dict],
    directions: list[str] | None,
    per_direction: int,
    candidate_pool: int,
    exclude_idxs: set[int],
) -> list[tuple[str, int]]:
    if items:
        rows = []
        for item in items:
            direction, value = item.split("=", 1)
            rows.append((direction, int(value)))
        return rows

    if directions:
        rows = []
        for direction in directions:
            candidates = [
                row for row in pair_rows
                if row["edit_direction"] == direction
                and row["selected_idx_orig"] != row["selected_idx_edit"]
                and int(row["dataset_idx"]) not in exclude_idxs
            ]
            if len(candidates) < candidate_pool:
                candidates = [
                    row for row in pair_rows
                    if row["edit_direction"] == direction
                    and int(row["dataset_idx"]) not in exclude_idxs
                ]
            candidates = sorted(
                candidates,
                key=lambda row: float(row["trajectory_l2_delta"]),
                reverse=True,
            )[:candidate_pool]
            rows.extend((direction, int(row["dataset_idx"])) for row in candidates)
        return rows

    return list(DEFAULT_ROWS)


def selected_pair(pair_rows: list[dict], direction: str, dataset_idx: int) -> dict:
    for row in pair_rows:
        if row["edit_direction"] == direction and int(row["dataset_idx"]) == dataset_idx:
            return row
    raise ValueError(f"No pair row for {direction} dataset_idx={dataset_idx}")


def selected_manifest(manifest_rows: list[dict], direction: str, dataset_idx: int) -> dict:
    for row in manifest_rows:
        if row.get("edit_direction") == direction and int(row.get("dataset_idx", -1)) == dataset_idx:
            return row
    raise ValueError(f"No manifest row for {direction} dataset_idx={dataset_idx}")


def load_sample_by_full_index(indexes: list[tuple[str, int, int]], data_dir: Path, full_idx: int) -> dict:
    filename, start_byte, byte_length = indexes[full_idx]
    frame = e2e_pb2.E2EDFrame()
    with (data_dir / filename).open("rb") as f:
        f.seek(start_byte)
        frame.ParseFromString(f.read(byte_length))

    past = np.stack(
        [
            frame.past_states.pos_x,
            frame.past_states.pos_y,
            frame.past_states.vel_x,
            frame.past_states.vel_y,
            frame.past_states.accel_x,
            frame.past_states.accel_y,
        ],
        axis=-1,
    ).astype(np.float32)
    future = np.stack([frame.future_states.pos_x, frame.future_states.pos_y], axis=-1).astype(np.float32)
    images_jpeg = [
        torch.from_numpy(np.frombuffer(img.image, dtype=np.uint8).copy())
        for img in frame.frame.images
    ]
    return {
        "PAST": past,
        "FUTURE": future,
        "IMAGES_JPEG": images_jpeg,
        "INTENT": frame.intent,
        "NAME": frame.frame.context.name,
    }


def select_traj(flat: torch.Tensor, scores: torch.Tensor, horizon: int) -> tuple[int, torch.Tensor]:
    scores = scores.float().view(-1)
    selected_idx = int(scores.argmin().item())
    traj = flat.float().view(scores.numel(), horizon, 2)[selected_idx]
    return selected_idx, traj


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    manifest_rows = read_jsonl(args.manifest)
    pair_rows = read_csv_dicts(args.pair_csv)
    exclude_idxs = excluded_dataset_idxs(args.exclude_cache)
    rows = sample_plan(
        args.samples,
        pair_rows=pair_rows,
        directions=args.directions,
        per_direction=args.per_direction,
        candidate_pool=args.candidate_pool,
        exclude_idxs=exclude_idxs,
    )

    with args.index_file.open("rb") as f:
        indexes = pickle.load(f)
    token_blob = torch.load(args.token_path, map_location="cpu")

    planner_model, lit_model = load_model(str(args.planner_checkpoint), device=device)
    planner_model.eval()

    out_rows = []
    kept_by_direction = {direction: 0 for direction, _ in rows}
    with torch.inference_mode():
        for direction, dataset_idx in rows:
            if args.expected_forward_delta and kept_by_direction.get(direction, 0) >= args.per_direction:
                continue
            pair = selected_pair(pair_rows, direction, dataset_idx)
            manifest = selected_manifest(manifest_rows, direction, dataset_idx)
            token_idx = int(pair["token_blob_idx"])
            camera_idx = int(pair.get("camera_idx", 1))
            sample = load_sample_by_full_index(indexes, args.data_dir, token_idx)
            if sample["NAME"] != pair["name"]:
                raise ValueError(f"Token/sample mismatch: {sample['NAME']} != {pair['name']}")

            edited_sample = {
                "PAST": sample["PAST"],
                "FUTURE": sample["FUTURE"],
                "INTENT": sample["INTENT"],
                "NAME": sample["NAME"],
                "IMAGES_JPEG": list(sample["IMAGES_JPEG"]),
            }
            edited_sample["IMAGES_JPEG"][camera_idx] = jpeg_bytes_to_tensor(Path(pair["edited_path"]))
            edited_batch = collate_with_images([edited_sample])
            edited_out = planner_model(
                model_inputs_from_batch(edited_batch, lit_model=lit_model, device=device),
                return_block_tokens=False,
            )

            horizon = int(token_blob["future"][token_idx].shape[0])
            orig_idx, orig_traj = select_traj(
                token_blob["trajectory"][token_idx],
                token_blob["scores"][token_idx],
                horizon,
            )
            edit_idx, edit_traj = select_traj(
                edited_out["trajectory"][0].detach().cpu(),
                edited_out["scores"][0].detach().cpu(),
                horizon,
            )
            future = token_blob["future"][token_idx].float()
            final_x_delta = float(edit_traj[-1, 0].item() - orig_traj[-1, 0].item())
            expected_sign = EXPECTED_FINAL_X_SIGN.get(direction)
            matches_expected = (
                expected_sign is None
                or final_x_delta == 0.0
                or final_x_delta * expected_sign > 0.0
            )
            if args.expected_forward_delta and not matches_expected:
                print(
                    f"skipped {direction} dataset_idx={dataset_idx}: "
                    f"final_x_delta={final_x_delta:+.2f}"
                )
                continue

            out_rows.append(
                {
                    "direction": direction,
                    "dataset_idx": dataset_idx,
                    "name": sample["NAME"],
                    "prompt": manifest.get("prompt", ""),
                    "edited_path": pair["edited_path"],
                    "token_blob_idx": token_idx,
                    "camera_idx": camera_idx,
                    "trajectory_l2_delta": float(pair["trajectory_l2_delta"]),
                    "latent_l2_delta": float(pair["latent_l2_delta"]),
                    "delta_selected_ade": float(pair["delta_selected_ade"]),
                    "final_x_delta": final_x_delta,
                    "expected_forward_delta": expected_sign,
                    "selected_idx_orig": orig_idx,
                    "selected_idx_edit": edit_idx,
                    "orig_traj": orig_traj.tolist(),
                    "edit_traj": edit_traj.tolist(),
                    "future": future.tolist(),
                }
            )
            kept_by_direction[direction] = kept_by_direction.get(direction, 0) + 1
            print(
                f"extracted {direction} dataset_idx={dataset_idx}: "
                f"{orig_idx}->{edit_idx}, final_x_delta={final_x_delta:+.2f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"samples": out_rows}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
