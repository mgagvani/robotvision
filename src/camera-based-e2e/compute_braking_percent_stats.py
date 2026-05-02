"""
Compute average braking percentage stats from an SAE intervention sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


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


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--exclude_object_name", type=str, default="traffic light")
    parser.add_argument("--front_camera_idx", type=int, default=1)
    parser.add_argument("--score_thresh", type=float, default=0.4)
    args = parser.parse_args()

    summary_path = resolve_input_path(args.summary_json, description="sweep summary JSON")
    results_path = resolve_input_path(args.results_csv, description="sweep results CSV")
    detections_path = resolve_input_path(args.detections, description="detections artifact")

    summary = json.loads(summary_path.read_text())
    best_setting = summary["best_setting"]["setting_name"]

    rows = [
        row for row in load_csv_rows(results_path)
        if row["setting_name"] == best_setting
    ]
    if not rows:
        raise RuntimeError(f"No rows found for best setting '{best_setting}'")

    artifact = torch.load(detections_path, map_location="cpu")
    categories = artifact["categories"]
    target_name = args.exclude_object_name.strip().lower()
    target_label_id = None
    for idx, category in enumerate(categories):
        if category.lower() == target_name:
            target_label_id = idx
            break
    if target_label_id is None:
        raise ValueError(f"Could not find detector category '{args.exclude_object_name}'")

    frame_to_has_object = {}
    for record in artifact["records"]:
        det = record.get("detections", {}).get(str(args.front_camera_idx))
        has_object = False
        if det is not None:
            for label, score in zip(det.get("labels", []), det.get("scores", [])):
                if int(label) == target_label_id and float(score) >= args.score_thresh:
                    has_object = True
                    break
        frame_to_has_object[record["frame_name"]] = has_object

    def pct_endpoint_decrease(row: dict) -> float:
        baseline = max(float(row["baseline_endpoint_distance_m"]), 1e-9)
        reduction = float(row["endpoint_distance_reduction_m"])
        return 100.0 * reduction / baseline

    def pct_path_decrease(row: dict) -> float:
        baseline = max(float(row["baseline_path_length_m"]), 1e-9)
        reduction = float(row["path_length_reduction_m"])
        return 100.0 * reduction / baseline

    all_endpoint_pct = [pct_endpoint_decrease(row) for row in rows]
    all_path_pct = [pct_path_decrease(row) for row in rows]
    filtered_rows = [
        row for row in rows
        if not frame_to_has_object.get(row["frame_name"], False)
    ]
    filtered_endpoint_pct = [pct_endpoint_decrease(row) for row in filtered_rows]
    filtered_path_pct = [pct_path_decrease(row) for row in filtered_rows]

    output = {
        "best_setting": best_setting,
        "excluded_object_name": args.exclude_object_name,
        "front_camera_idx": args.front_camera_idx,
        "score_thresh": args.score_thresh,
        "all_frames": {
            "n": len(rows),
            "mean_pct_endpoint_decrease": mean(all_endpoint_pct),
            "mean_pct_path_decrease": mean(all_path_pct),
            "mean_endpoint_distance_reduction_m": mean(
                [float(row["endpoint_distance_reduction_m"]) for row in rows]
            ),
        },
        "excluding_object_frames": {
            "n": len(filtered_rows),
            "mean_pct_endpoint_decrease": mean(filtered_endpoint_pct),
            "mean_pct_path_decrease": mean(filtered_path_pct),
            "mean_endpoint_distance_reduction_m": mean(
                [float(row["endpoint_distance_reduction_m"]) for row in filtered_rows]
            ),
        },
        "excluded_frame_count": len(rows) - len(filtered_rows),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
