"""Quick coordinate consistency check.

Verifies that the preprocessing pipeline (point_cloud_gpu.py) and the
model's DepthGuidedLift produce identical vehicle-frame point coordinates
for the same input frame.

Run on a GPU node:
    python test_coord_consistency.py \
        --data_dir /scratch/gilbreth/kumar753/robotvision/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
        --index_pkl /scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e/index_train.pkl
"""

import argparse
import pickle
import os
import numpy as np
import torch
import cv2

from protos import e2e_pb2
from point_cloud_gpu import (
    backproject_depth_to_points,
    points_to_vehicle_frame,
    undistort_image,
)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F


def get_depth_maps(rgb_images, depth_model, depth_processor, device, max_depth=50.0):
    original_sizes = [(img.shape[0], img.shape[1]) for img in rgb_images]
    max_h = max(s[0] for s in original_sizes)
    max_w = max(s[1] for s in original_sizes)
    padded = []
    for img in rgb_images:
        h, w = img.shape[:2]
        p = cv2.copyMakeBorder(img, 0, max_h-h, 0, max_w-w, cv2.BORDER_REPLICATE)
        padded.append(p)
    inputs = depth_processor(images=padded, return_tensors="pt").to(device)
    with torch.no_grad():
        preds = depth_model(**inputs).predicted_depth
    depth_maps = []
    for i, (orig_h, orig_w) in enumerate(original_sizes):
        pred_i = preds[i:i+1].unsqueeze(1).float()
        pred_full = F.interpolate(pred_i, size=(max_h, max_w), mode="bicubic", align_corners=False).squeeze()
        depth_maps.append(pred_full[:orig_h, :orig_w].clamp(0.1, max_depth))
    return depth_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--index_pkl", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load one frame
    with open(args.index_pkl, "rb") as f:
        indexes = pickle.load(f)
    filename, start_byte, byte_length = indexes[args.frame_idx]
    with open(os.path.join(args.data_dir, filename), "rb") as f:
        f.seek(start_byte)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(f.read(byte_length))
    print(f"Loaded frame {args.frame_idx} from {filename}")

    # Load depth model
    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
    depth_model.eval()
    if device.type == "cuda":
        depth_model = depth_model.half()

    # Extract calibrations and images
    calib_by_name = {}
    for calib in frame.frame.context.camera_calibrations:
        intr = list(calib.intrinsic)
        extr = np.array(list(calib.extrinsic.transform), dtype=np.float64).reshape(4, 4)
        calib_by_name[calib.name] = {"intrinsic": intr, "extrinsic": extr,
                                      "width": calib.width, "height": calib.height}

    rgb_images, intrinsics_list, extrinsics_list = [], [], []
    for img_proto in frame.frame.images:
        cam_name = img_proto.name
        if cam_name not in calib_by_name:
            continue
        cal = calib_by_name[cam_name]
        jpg = np.frombuffer(img_proto.image, dtype=np.uint8)
        rgb = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        intr = cal["intrinsic"]
        intrinsics_list.append([intr[0], intr[1], intr[2], intr[3], cal["width"], cal["height"]] + intr[4:9])
        extrinsics_list.append(cal["extrinsic"])
        rgb_images.append(rgb)

    # Get depth maps
    depth_maps = get_depth_maps(rgb_images, depth_model, depth_processor, device)

    # ── Method 1: preprocessing pipeline (point_cloud_gpu.py) ──
    pts_method1 = []
    for i, (depth_gpu, intr_vals, extr_np) in enumerate(zip(depth_maps, intrinsics_list, extrinsics_list)):
        fx, fy, cx, cy = intr_vals[0], intr_vals[1], intr_vals[2], intr_vals[3]
        if len(intr_vals) > 6:
            dist = intr_vals[6:11]
            depth_np = depth_gpu.cpu().numpy()
            depth_np = undistort_image(depth_np, fx, fy, cx, cy, dist)
            depth_gpu = torch.from_numpy(depth_np).to(device)
        pts_cv = backproject_depth_to_points(depth_gpu, fx, fy, cx, cy)
        d_flat = depth_gpu.reshape(-1)
        valid = (d_flat > 0.1) & (d_flat < 50.0)
        pts_cv = pts_cv[valid]
        extr_gpu = torch.from_numpy(extr_np.astype(np.float32)).to(device)
        pts_veh = points_to_vehicle_frame(pts_cv, extr_gpu, device)
        pts_method1.append(pts_veh)

    # ── Method 2: DepthGuidedLift logic ──
    T_cv = torch.tensor(
        [[ 0., 0., 1., 0.],
         [-1., 0., 0., 0.],
         [ 0.,-1., 0., 0.],
         [ 0., 0., 0., 1.]], device=device, dtype=torch.float32
    )

    pts_method2 = []
    for i, (depth_gpu, intr_vals, extr_np) in enumerate(zip(depth_maps, intrinsics_list, extrinsics_list)):
        fx, fy, cx, cy = intr_vals[0], intr_vals[1], intr_vals[2], intr_vals[3]
        if len(intr_vals) > 6:
            dist = intr_vals[6:11]
            depth_np = depth_gpu.cpu().numpy()
            depth_np = undistort_image(depth_np, fx, fy, cx, cy, dist)
            depth_gpu = torch.from_numpy(depth_np).to(device)

        H, W = depth_gpu.shape
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        d = depth_gpu.reshape(-1)
        u = u_grid.reshape(-1)
        v = v_grid.reshape(-1)
        valid = (d > 0.1) & (d < 50.0)
        d, u, v = d[valid], u[valid], v[valid]

        X_c = (u - cx) * d / fx
        Y_c = (v - cy) * d / fy
        Z_c = d
        ones = torch.ones_like(d)
        pts_cam = torch.stack([X_c, Y_c, Z_c, ones], dim=0)  # (4, N)

        extr_gpu = torch.from_numpy(extr_np.astype(np.float32)).to(device)
        pts_veh = extr_gpu @ (T_cv @ pts_cam)  # (4, N)
        pts_method2.append(pts_veh[:3].T)  # (N, 3)

    # ── Compare ──
    print("\n=== Coordinate Consistency Check ===")
    all_ok = True
    for i, (p1, p2) in enumerate(zip(pts_method1, pts_method2)):
        n = min(p1.shape[0], p2.shape[0])
        diff = (p1[:n] - p2[:n]).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        ok = max_err < 0.01
        status = "OK" if ok else "MISMATCH"
        print(f"  Camera {i}: max_err={max_err:.6f}  mean_err={mean_err:.6f}  [{status}]")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("PASS — coordinate frames are consistent.")
    else:
        print("FAIL — coordinate mismatch detected! BEV and OCC labels will be misaligned.")


if __name__ == "__main__":
    main()
