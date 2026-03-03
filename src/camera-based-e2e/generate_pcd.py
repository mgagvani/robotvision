import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torchvision
import torchvision.io

# add this file's folder so local imports work
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from depthLoss import DepthLoss
from point_cloud import create_point_cloud
from loader import WaymoE2E, collate_fn


# map waymo enum → readable camera name
CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}


def get_waymo_intrinsics(frame_proto, camera_name_enum):
    # grab intrinsics for one camera from the frame proto
    for calib in frame_proto.context.camera_calibrations:
        if calib.name == camera_name_enum:
            fx, fy, cx, cy = (
                calib.intrinsic[0],
                calib.intrinsic[1],
                calib.intrinsic[2],
                calib.intrinsic[3],
            )

            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(calib.width, calib.height, fx, fy, cx, cy)
            return intrinsics

    raise ValueError(f"camera {camera_name_enum} not found")


def get_extrinsic_matrix(frame_proto, camera_name_enum):
    # camera → vehicle transform (4x4)
    for calib in frame_proto.context.camera_calibrations:
        if calib.name == camera_name_enum:
            return np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)

    raise ValueError(f"camera {camera_name_enum} not found")


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    d_min, d_max = depth.min(), depth.max()

    if d_max - d_min < 1e-6:
        return np.ones_like(depth) * 10.0

    depth_inv = 1.0 - (depth - d_min) / (d_max - d_min)
    metric_depth = 0.5 + depth_inv * (80.0 - 0.5)

    return metric_depth.astype(np.float32)


def reconstruct_scene_from_frame(
    frame_proto,
    jpeg_tensors_for_frame,
    depth_model: DepthLoss,
    device: torch.device,
) -> o3d.geometry.PointCloud:

    # build one big point cloud by fusing all 5 cameras
    all_pcds = []

    for cam_idx, cam_image_proto in enumerate(frame_proto.images):
        cam_enum = cam_image_proto.name
        cam_name = CAMERA_NAMES.get(cam_enum, f"CAM_{cam_enum}")

        # decode jpeg straight onto gpu
        jpeg_bytes = jpeg_tensors_for_frame[cam_idx]

        image_tensor = torchvision.io.decode_jpeg(
            jpeg_bytes,
            mode=torchvision.io.ImageReadMode.RGB,
            device=device,
        )

        image_batch = image_tensor.unsqueeze(0).float()

        # run depth model
        with torch.no_grad():
            depth = depth_model.get_depth(image_batch)[0].cpu().numpy()

        rgb_np = image_tensor.cpu().permute(1, 2, 0).byte().numpy().copy()

        # get camera calibration
        intrinsics = get_waymo_intrinsics(frame_proto, cam_enum)
        extrinsic = get_extrinsic_matrix(frame_proto, cam_enum)

        depth_metric = normalize_depth(depth)

        # pixels → 3d points (camera space)
        pcd = create_point_cloud(rgb_np, depth_metric, intrinsics=intrinsics)

        # move into vehicle frame
        pcd.transform(extrinsic)

        all_pcds.append(pcd)
        print(f"    {cam_name:>12s}: {len(pcd.points):>8,} points")

    # merge all camera clouds together
    merged = o3d.geometry.PointCloud()
    for pcd in all_pcds:
        merged += pcd

    return merged


def save_merged_pcd(
    merged_pcd: o3d.geometry.PointCloud,
    output_dir: Path,
    frame_idx: int,
    voxel_size: float = 0.05,
) -> Path:

    # save a single fused cloud (optionally downsample first)
    output_dir.mkdir(parents=True, exist_ok=True)

    if voxel_size > 0:
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)

    filename = output_dir / f"scene_frame{frame_idx:04d}.pcd"

    o3d.io.write_point_cloud(str(filename), merged_pcd)

    print(
        f"  Saved {filename.name}  "
        f"({len(merged_pcd.points):,} points after voxel={voxel_size}m downsampling)"
    )

    return filename


def main():
    parser = argparse.ArgumentParser(
        description="generate merged 5-camera pcd files from waymo frames"
    )

    parser.add_argument("--index_file", default="index_val.pkl")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./pcd_output")
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--skip_batches", type=int, default=0)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    # pick gpu if available
    if not torch.cuda.is_available():
        print("warning: cuda not available — cpu will be slow")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("building dataset…")

    dataset = WaymoE2E(
        indexFile=args.index_file,
        data_dir=args.data_dir,
    )

    data_iter = iter(torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    ))

    print("loading depth-anything-v2…")
    depth_model = DepthLoss(device)
    print("depth model ready\n")

    output_dir = Path(args.output_dir)

    # skip some batches if requested
    for _ in range(args.skip_batches):
        next(data_iter)

    global_frame_idx = 0

    # main loop over batches + frames
    for batch_idx in range(args.num_batches):
        print(f"─── Batch {batch_idx + 1}/{args.num_batches} ───")

        batch = next(data_iter)

        frames = batch["FRAME"]
        jpegs = batch["IMAGES_JPEG"]

        for i, frame_proto in enumerate(frames):
            print(f"  Frame {i} (global {global_frame_idx})")

            # collect jpeg tensors for this frame across cameras
            jpeg_tensors_for_frame = [
                jpegs[cam_idx][i]
                for cam_idx in range(len(jpegs))
            ]

            # currently the individual frames look fine
            # merged scenes look weird so need to fix depth model
            merged_pcd = reconstruct_scene_from_frame(
                frame_proto=frame_proto,
                jpeg_tensors_for_frame=jpeg_tensors_for_frame,
                depth_model=depth_model,
                device=device,
            )

            save_merged_pcd(
                merged_pcd=merged_pcd,
                output_dir=output_dir,
                frame_idx=global_frame_idx,
                voxel_size=args.voxel_size,
            )

            global_frame_idx += 1

    print(f"\ndone. {global_frame_idx} pcd files written to {output_dir.resolve()}")
    print("view with: python view_pcd.py <file>.pcd or cloudcompare")


if __name__ == "__main__":
    main()
