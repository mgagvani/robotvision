import argparse
import json
import pickle
import re
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from protos import dataset_pb2, e2e_pb2, map_pb2


def safe_name(name: str) -> str:
    # Keep filesystem-friendly ASCII.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def camera_name(name_enum: int) -> str:
    return dataset_pb2.CameraName.Name.Name(name_enum)


def intent_name(intent_enum: int) -> str:
    return e2e_pb2.EgoIntent.Intent.Name(intent_enum)


def transform_list(transform_msg) -> list:
    return list(transform_msg.transform)


def camera_calibration_dict(calib) -> Dict[str, object]:
    return {
        "name": camera_name(calib.name),
        "intrinsic": list(calib.intrinsic),
        "extrinsic": transform_list(calib.extrinsic),
        "width": calib.width,
        "height": calib.height,
        "rolling_shutter_direction": dataset_pb2.CameraCalibration.RollingShutterReadOutDirection.Name(
            calib.rolling_shutter_direction
        ),
    }


def camera_image_meta(img, image_path: str) -> Dict[str, object]:
    return {
        "file": image_path,
        "pose": transform_list(img.pose),
        "velocity": {
            "v_x": img.velocity.v_x,
            "v_y": img.velocity.v_y,
            "v_z": img.velocity.v_z,
            "w_x": img.velocity.w_x,
            "w_y": img.velocity.w_y,
            "w_z": img.velocity.w_z,
        },
        "pose_timestamp": img.pose_timestamp,
        "shutter": img.shutter,
        "camera_trigger_time": img.camera_trigger_time,
        "camera_readout_done_time": img.camera_readout_done_time,
    }


def extract_states(frame: e2e_pb2.E2EDFrame) -> Tuple[np.ndarray, np.ndarray]:
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

    if len(frame.future_states.pos_x):
        future = np.stack(
            [frame.future_states.pos_x, frame.future_states.pos_y], axis=-1
        ).astype(np.float32)
    else:
        future = np.zeros((0, 2), dtype=np.float32)

    return past, future


def iter_indexed_frames(
    index: Iterable[Tuple[str, int, int]], data_dir: Path
) -> Iterator[Tuple[e2e_pb2.E2EDFrame, str, int, int]]:
    current_file = None
    current_filename = None

    for filename, start_byte, byte_length in index:
        if filename != current_filename:
            if current_file:
                current_file.close()
            current_file = (data_dir / filename).open("rb")
            current_filename = filename

        current_file.seek(start_byte)
        protobuf = current_file.read(byte_length)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(protobuf)

        yield frame, filename, start_byte, byte_length

    if current_file:
        current_file.close()


def write_json(path: Path, data: Dict[str, object], pretty: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=True,
            indent=2 if pretty else None,
            sort_keys=pretty,
        )


def shard_dir(base: Path, sample_id: str, shard_len: int) -> Path:
    if shard_len <= 0:
        return base / sample_id
    shard = sample_id[:shard_len]
    return base / shard / sample_id


def make_sample_id(
    raw_id: str,
    global_index: int,
    id_mode: str,
    id_counts: Optional[Dict[str, int]],
) -> str:
    base = safe_name(raw_id) if raw_id else f"idx_{global_index}"
    if id_mode == "context_index":
        return f"{base}__{global_index}"
    if id_counts is None:
        return base
    count = id_counts.get(base, 0)
    id_counts[base] = count + 1
    if count:
        return f"{base}_{count}"
    return base


def export_frame(
    frame: e2e_pb2.E2EDFrame,
    filename: str,
    start_byte: int,
    byte_length: int,
    global_index: int,
    cfg: Dict[str, object],
    id_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    raw_id = frame.frame.context.name or f"idx_{global_index}"
    sample_id = make_sample_id(raw_id, global_index, cfg["id_mode"], id_counts)

    samples_dir: Path = cfg["samples_dir"]
    split_dir: Path = cfg["split_dir"]
    shard_len: int = cfg["shard_len"]
    image_ext: str = cfg["image_ext"]
    write_images: bool = cfg["write_images"]
    write_calib: bool = cfg["write_calib"]
    write_map_features: bool = cfg["write_map_features"]
    pretty_json: bool = cfg["pretty_json"]
    overwrite: bool = cfg["overwrite"]

    sample_dir = shard_dir(samples_dir, sample_id, shard_len)
    images_dir = sample_dir / "images"
    should_write = overwrite or not (sample_dir / "meta.json").exists()

    if should_write:
        sample_dir.mkdir(parents=True, exist_ok=True)
        if write_images:
            images_dir.mkdir(parents=True, exist_ok=True)

    past, future = extract_states(frame)
    state_path = sample_dir / "state.npz"
    if should_write:
        np.savez_compressed(
            state_path,
            past=past,
            future=future,
            intent=np.int32(frame.intent),
        )

    images_meta: Dict[str, object] = {}
    if frame.frame.images:
        for img in frame.frame.images:
            cam = camera_name(img.name)
            image_rel = Path("images") / f"{cam}.{image_ext}"
            image_path = sample_dir / image_rel
            if should_write and write_images:
                with image_path.open("wb") as f:
                    f.write(img.image)
            images_meta[cam] = camera_image_meta(img, image_rel.as_posix())

    calib_path = sample_dir / "calib.json"
    if should_write and write_calib:
        calib = [
            camera_calibration_dict(c)
            for c in frame.frame.context.camera_calibrations
        ]
        write_json(calib_path, {"cameras": calib}, pretty_json)

    map_features_path = None
    if write_map_features and frame.frame.map_features:
        map_features_path = sample_dir / "map_features.pb"
        if should_write:
            map_msg = map_pb2.Map()
            map_msg.map_features.extend(frame.frame.map_features)
            with map_features_path.open("wb") as f:
                f.write(map_msg.SerializeToString())

    meta = {
        "id": sample_id,
        "context_name": frame.frame.context.name,
        "timestamp_micros": frame.frame.timestamp_micros,
        "intent": {"id": frame.intent, "name": intent_name(frame.intent)},
        "source": {
            "tfrecord": filename,
            "start_byte": start_byte,
            "byte_length": byte_length,
            "index": global_index,
        },
        "frame_pose": transform_list(frame.frame.pose)
        if frame.frame.HasField("pose")
        else [],
        "map_pose_offset": [
            frame.frame.map_pose_offset.x,
            frame.frame.map_pose_offset.y,
            frame.frame.map_pose_offset.z,
        ]
        if frame.frame.HasField("map_pose_offset")
        else None,
        "images": images_meta,
        "calib_file": "calib.json" if write_calib else None,
        "map_features_file": "map_features.pb" if map_features_path else None,
        "extras": {
            "depth": None,
            "hd_map": None,
            "bev": None,
        },
    }

    meta_path = sample_dir / "meta.json"
    if should_write:
        write_json(meta_path, meta, pretty_json)

    rel_sample_dir = sample_dir.relative_to(split_dir)
    entry = {
        "id": sample_id,
        "sample_dir": rel_sample_dir.as_posix(),
        "state": str(state_path.relative_to(split_dir)),
        "meta": str(meta_path.relative_to(split_dir)),
        "images": {
            cam: str(
                (sample_dir / Path(info["file"])).relative_to(split_dir)
            )
            for cam, info in images_meta.items()
        },
    }
    return entry


def chunk_tasks(tasks: List[Tuple[int, str, int, int]], chunk_size: int) -> List[List[Tuple[int, str, int, int]]]:
    return [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]


def export_chunk(
    payload: Tuple[int, List[Tuple[int, str, int, int]]],
    cfg: Dict[str, object],
) -> str:
    chunk_id, chunk = payload
    data_dir: Path = cfg["data_dir"]
    split_dir: Path = cfg["split_dir"]
    manifest_part = split_dir / f"manifest.part_{chunk_id:09d}.jsonl"

    current_file = None
    current_filename = None

    with manifest_part.open("w", encoding="utf-8") as manifest_file:
        for global_index, filename, start_byte, byte_length in chunk:
            if filename != current_filename:
                if current_file:
                    current_file.close()
                current_file = (data_dir / filename).open("rb")
                current_filename = filename

            current_file.seek(start_byte)
            protobuf = current_file.read(byte_length)
            frame = e2e_pb2.E2EDFrame()
            frame.ParseFromString(protobuf)

            entry = export_frame(
                frame,
                filename,
                start_byte,
                byte_length,
                global_index,
                cfg,
                id_counts=None,
            )
            manifest_file.write(json.dumps(entry, ensure_ascii=True) + "\n")

    if current_file:
        current_file.close()

    return manifest_part.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Waymo E2E TFRecord-indexed data into a folder-based layout."
    )
    parser.add_argument(
        "--index-file",
        required=True,
        help="Path to index_*.pkl with (filename, start_byte, byte_length).",
    )
    parser.add_argument("--data-dir", required=True, help="TFRecord data directory.")
    parser.add_argument("--out-dir", required=True, help="Output dataset root.")
    parser.add_argument("--split", required=True, help="Split name (train/val/test).")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to export.",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        help="File extension for images (default: jpg).",
    )
    parser.add_argument(
        "--no-write-images",
        action="store_true",
        help="Skip writing image files (metadata only).",
    )
    parser.add_argument(
        "--no-write-calib",
        action="store_true",
        help="Skip writing camera calibration files.",
    )
    parser.add_argument(
        "--write-map-features",
        action="store_true",
        help="Write per-sample map features if present.",
    )
    parser.add_argument(
        "--shard-len",
        type=int,
        default=2,
        help="Prefix length for sharded sample folders (0 to disable).",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Pretty-print JSON outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sample outputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes (1 = single-process).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Samples per worker chunk for parallel export.",
    )
    parser.add_argument(
        "--id-mode",
        choices=["context", "context_index"],
        default="context",
        help="Sample id mode: context (default) or context_index for parallel safety.",
    )
    parser.add_argument(
        "--delete-manifest-parts",
        action="store_true",
        help="Delete per-chunk manifest.part_* files after merge.",
    )

    args = parser.parse_args()

    index_path = Path(args.index_file)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    split_dir = out_dir / args.split
    samples_dir = split_dir / "samples"
    manifest_path = split_dir / "manifest.jsonl"

    split_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    with index_path.open("rb") as f:
        index = pickle.load(f)

    start_idx = max(args.start_idx, 0)
    end_idx = len(index)
    if args.max_samples is not None:
        end_idx = min(end_idx, start_idx + args.max_samples)
    index_slice = index[start_idx:end_idx]

    write_images = not args.no_write_images
    write_calib = not args.no_write_calib
    write_map_features = args.write_map_features
    pretty_json = args.pretty_json
    id_mode = args.id_mode

    if args.num_workers > 1 and id_mode == "context":
        print(
            "Parallel export requires unique ids across workers; "
            "switching --id-mode to context_index."
        )
        id_mode = "context_index"

    cfg: Dict[str, object] = {
        "data_dir": data_dir,
        "split_dir": split_dir,
        "samples_dir": samples_dir,
        "shard_len": args.shard_len,
        "image_ext": args.image_ext,
        "write_images": write_images,
        "write_calib": write_calib,
        "write_map_features": write_map_features,
        "pretty_json": pretty_json,
        "overwrite": args.overwrite,
        "id_mode": id_mode,
    }

    if args.num_workers <= 1:
        id_counts = {} if id_mode == "context" else None
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for idx_offset, (frame, filename, start_byte, byte_length) in enumerate(
                tqdm(iter_indexed_frames(index_slice, data_dir), total=len(index_slice))
            ):
                global_index = start_idx + idx_offset
                entry = export_frame(
                    frame,
                    filename,
                    start_byte,
                    byte_length,
                    global_index,
                    cfg,
                    id_counts=id_counts,
                )
                manifest_file.write(json.dumps(entry, ensure_ascii=True) + "\n")
    else:
        tasks = [
            (start_idx + i, filename, start_byte, byte_length)
            for i, (filename, start_byte, byte_length) in enumerate(index_slice)
        ]
        chunks = chunk_tasks(tasks, args.chunk_size)
        payloads = [(chunk[0][0], chunk) for chunk in chunks if chunk]

        worker = partial(export_chunk, cfg=cfg)
        part_paths = process_map(
            worker,
            payloads,
            max_workers=args.num_workers,
            chunksize=1,
            total=len(payloads),
        )

        part_paths = sorted(part_paths)
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for part in part_paths:
                part_path = Path(part)
                with part_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        manifest_file.write(line)

        if args.delete_manifest_parts:
            for part in part_paths:
                try:
                    Path(part).unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
