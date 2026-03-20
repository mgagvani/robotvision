#!/usr/bin/env python3
"""Convert indexed Waymo E2E tfrecords into .jpg + .json.gz pairs
for the WODE2EData training loader.

Output layout (one pair per camera per frame):
  out_dir/
    {global_index // 1000:04d}/
      {global_index:07d}_{CAMERA_NAME}.jpg
      {global_index:07d}_{CAMERA_NAME}.json.gz

The .json.gz format expected by WODE2EData._load():
  {
    "past":       {"x": [...], "y": [...], "vx": [...], "vy": [...]},
    "command":    int,          # frame.intent
    "prediction": {"x": [...], "y": [...]}   # future states (omitted if empty)
  }
"""
import argparse
import gzip
import json
import pickle
import sys
from functools import partial
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent))
from protos import dataset_pb2, e2e_pb2


def camera_name(name_enum: int) -> str:
    return dataset_pb2.CameraName.Name.Name(name_enum)


def process_chunk(payload, cfg):
    chunk_id, chunk = payload
    data_dir: Path = cfg["data_dir"]
    out_dir: Path = cfg["out_dir"]
    cameras: set = cfg["cameras"]

    current_file = None
    current_filename = None

    for global_index, filename, start_byte, byte_length in chunk:
        if filename != current_filename:
            if current_file:
                current_file.close()
            current_file = (data_dir / filename).open("rb")
            current_filename = filename

        current_file.seek(start_byte)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(current_file.read(byte_length))

        # Ego state — shared across all cameras of this frame
        measurement = {
            "past": {
                "x": list(frame.past_states.pos_x),
                "y": list(frame.past_states.pos_y),
                "vx": list(frame.past_states.vel_x),
                "vy": list(frame.past_states.vel_y),
            },
            "command": int(frame.intent),
        }
        if len(frame.future_states.pos_x) > 0:
            measurement["prediction"] = {
                "x": list(frame.future_states.pos_x),
                "y": list(frame.future_states.pos_y),
            }
        if len(frame.preference_trajectories) > 0:
            measurement["preferences"] = [
                {
                    "pos_x": list(pt.pos_x),
                    "pos_y": list(pt.pos_y),
                    "preference_score": float(pt.preference_score),
                }
                for pt in frame.preference_trajectories
            ]

        shard_dir = out_dir / f"{global_index // 1000:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        for img in frame.frame.images:
            cam = camera_name(img.name)
            if cameras and cam not in cameras:
                continue

            base = f"{global_index:07d}_{cam}"
            jpg_path = shard_dir / f"{base}.jpg"
            meta_path = shard_dir / f"{base}.json.gz"

            if not jpg_path.exists():
                jpg_path.write_bytes(img.image)
            if not meta_path.exists():
                with gzip.open(meta_path, "wt", encoding="utf-8") as f:
                    json.dump(measurement, f)

    if current_file:
        current_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert indexed Waymo E2E tfrecords to .jpg + .json.gz pairs."
    )
    parser.add_argument("--index-file", required=True, help="Path to index_*.pkl")
    parser.add_argument("--data-dir", required=True, help="TFRecord directory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
        help="Camera names to extract (default: 3 front cameras)",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.index_file, "rb") as f:
        index = pickle.load(f)

    start = args.start_idx
    end = len(index) if args.max_samples is None else min(len(index), start + args.max_samples)
    index_slice = index[start:end]

    print(f"Processing {len(index_slice)} frames x {len(args.cameras)} cameras -> {out_dir}")

    cfg = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "cameras": set(args.cameras),
    }

    tasks = [
        (start + i, filename, sb, bl)
        for i, (filename, sb, bl) in enumerate(index_slice)
    ]
    chunks = [tasks[i : i + args.chunk_size] for i in range(0, len(tasks), args.chunk_size)]
    payloads = [(i, chunk) for i, chunk in enumerate(chunks) if chunk]

    worker = partial(process_chunk, cfg=cfg)
    if args.num_workers <= 1:
        for payload in tqdm(payloads, desc="chunks"):
            worker(payload)
    else:
        process_map(
            worker, payloads, max_workers=args.num_workers, chunksize=1, total=len(payloads)
        )

    print(f"Done. Output written to: {out_dir}")


if __name__ == "__main__":
    main()
