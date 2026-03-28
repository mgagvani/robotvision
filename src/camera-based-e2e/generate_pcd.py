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
from point_cloud import create_point_cloud, transform_to_vehicle_frame, merge_point_clouds, T_WAYMO_TO_O3D
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
    for calib in frame_proto.context.camera_calibrations:
        if calib.name == camera_name_enum:
            return np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
    raise ValueError(f"camera {camera_name_enum} not found")


def reconstruct_scene_from_frame(
    frame_proto,
    jpeg_tensors_for_frame,
    depth_model: DepthLoss,
    device: torch.device,
    output_dir: Path,
    frame_idx: int,
) -> o3d.geometry.PointCloud:

    output_dir.mkdir(parents=True, exist_ok=True)
    all_pcds = []

    for cam_idx, cam_image_proto in enumerate(frame_proto.images):
        cam_enum = cam_image_proto.name
        cam_name = CAMERA_NAMES.get(cam_enum, f"CAM_{cam_enum}")

        jpeg_bytes = jpeg_tensors_for_frame[cam_idx]
        image_tensor = torchvision.io.decode_jpeg(
            jpeg_bytes,
            mode=torchvision.io.ImageReadMode.RGB,
            device=device,
        )
        H = image_tensor.shape[1]
        crop_start = int(H * 0.30)
        image_tensor = image_tensor[:, crop_start:, :]
        image_batch = image_tensor.unsqueeze(0).float()

        o3d_intrinsics = get_waymo_intrinsics(frame_proto, cam_enum)
        fx = o3d_intrinsics.intrinsic_matrix[0, 0]
        fy = o3d_intrinsics.intrinsic_matrix[1, 1]
        cx = o3d_intrinsics.intrinsic_matrix[0, 2]
        cy = o3d_intrinsics.intrinsic_matrix[1, 2] - crop_start

        new_H = image_tensor.shape[1]
        o3d_intrinsics.set_intrinsics(o3d_intrinsics.width, new_H, fx, fy, cx, cy)

        K = torch.tensor([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            depth = depth_model.get_depth(image_batch, intrinsics=K)[0].cpu().numpy().squeeze()
            depth = np.clip(depth, 0.5, 50.0)
            rgb_np = image_tensor.cpu().permute(1, 2, 0).byte().numpy().copy()
            R, G, B = rgb_np[:,:,0].astype(float), rgb_np[:,:,1].astype(float), rgb_np[:,:,2].astype(float)
            sky_mask = (B > 120) & (B > R*1.2) & (B > G*1.1) & (depth > 40)
            depth[sky_mask] = 0.0  # open3d skips zero-depth pixels

        extrinsic = get_extrinsic_matrix(frame_proto, cam_enum)

        # back-project to point cloud (OpenCV frame, no flip)
        pcd = create_point_cloud(rgb_np, depth, o3d_intrinsics)

        # OpenCV -> Waymo camera -> vehicle frame (correct coordinate transforms)
        transform_to_vehicle_frame(pcd, extrinsic)

        # save individual camera cloud
        cam_pcd = o3d.geometry.PointCloud(pcd)
        cam_pcd.transform(T_WAYMO_TO_O3D)
        cam_filename = output_dir / f"cam_{cam_name}_frame{frame_idx:04d}.pcd"
        o3d.io.write_point_cloud(str(cam_filename), cam_pcd)

        all_pcds.append(pcd)
        print(f"    {cam_name:>12s}: {len(pcd.points):>8,} points → {cam_filename.name}")

    merged = merge_point_clouds(all_pcds)

    # remove outlier points
    # if len(merged.points) > 0:
      #  merged, _ = merged.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
       # merged, _ = merged.remove_radius_outlier(nb_points=10, radius=0.5)

    return merged


def save_merged_pcd(
    merged_pcd: o3d.geometry.PointCloud,
    output_dir: Path,
    frame_idx: int,
    voxel_size: float = 0.05,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)

    if voxel_size > 0:
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
    merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
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

    print("loading unidepth-v2…")
    depth_model = DepthLoss(device)
    print("depth model ready\n")

    output_dir = Path(args.output_dir)

    for _ in range(args.skip_batches):
        next(data_iter)

    global_frame_idx = 0

    for batch_idx in range(args.num_batches):
        print(f"─── Batch {batch_idx + 1}/{args.num_batches} ───")

        batch = next(data_iter)
        frames = batch["FRAME"]
        jpegs = batch["IMAGES_JPEG"]

        for i, frame_proto in enumerate(frames):
            print(f"  Frame {i} (global {global_frame_idx})")

            jpeg_tensors_for_frame = [
                jpegs[cam_idx][i]
                for cam_idx in range(len(jpegs))
            ]

            merged_pcd = reconstruct_scene_from_frame(
                frame_proto=frame_proto,
                jpeg_tensors_for_frame=jpeg_tensors_for_frame,
                depth_model=depth_model,
                device=device,
                output_dir=output_dir,
                frame_idx=global_frame_idx,
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
