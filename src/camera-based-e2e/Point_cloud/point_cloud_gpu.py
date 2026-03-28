"""GPU-accelerated BEV generation pipeline.

Pure PyTorch replacement for point_cloud.py — all geometry operations run on
GPU tensors so the GPU never sits idle between depth inference and BEV output.

Key differences from the CPU (Open3D) pipeline:
  • Back-projection uses vectorised meshgrid + intrinsics (no Open3D)
  • Coordinate transforms via batched torch.matmul
  • No disparity-to-depth conversion (Depth-Anything-V2 outputs depth directly)
  • No metric-scale correction (not needed with direct depth)
  • Outlier removal replaced by simple range-based clipping
  • BEV rasterisation uses scatter_reduce on GPU
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from protos import dataset_pb2  # Standard waymo import

# ── Constant transforms (as float32 GPU tensors — lazily moved to device) ────

# OpenCV camera frame (+X right, +Y down, +Z forward)
# → Waymo camera frame (+X forward, +Y left, +Z up)
_T_CV_TO_WAYMO = torch.tensor(
    [[ 0.,  0.,  1.,  0.],
     [-1.,  0.,  0.,  0.],
     [ 0., -1.,  0.,  0.],
     [ 0.,  0.,  0.,  1.]],
    dtype=torch.float32,
)

# ── BEV grid parameters ─────────────────────────────────────────────────────
BEV_RANGE = 25.0     # metres in each direction from ego
BEV_RES   = 0.25     # metres per pixel
BEV_SIZE  = int(2 * BEV_RANGE / BEV_RES)  # 200


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a constant tensor to *device* on first use, then cache."""
    if t.device != device:
        return t.to(device)
    return t


def undistort_image(image, fx, fy, cx, cy, dist_coeffs):
    """Removes lens distortion (OpenCV, CPU).  Fast enough to keep on CPU."""
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    return cv2.undistort(image, camera_matrix, dist)


# ── Core GPU operations ─────────────────────────────────────────────────────

def backproject_depth_to_points(
    depth: torch.Tensor,
    fx: float, fy: float,
    cx: float, cy: float,
) -> torch.Tensor:
    """Back-project a depth map to 3D points in OpenCV camera frame.

    Args:
        depth: (H, W) float tensor — metric depth in metres.
        fx, fy, cx, cy: pinhole intrinsics.

    Returns:
        points: (H*W, 3) float tensor in OpenCV camera coords.
    """
    H, W = depth.shape
    device = depth.device

    # Pixel coordinate grids — (H, W) each
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Flatten
    u = u.reshape(-1)   # (N,)
    v = v.reshape(-1)
    d = depth.reshape(-1)  # (N,)

    # Back-project: X = (u - cx) * d / fx  etc.
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    return torch.stack([X, Y, Z], dim=-1)  # (N, 3)


def transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a 4×4 rigid transform to (N, 3) points.

    Args:
        points: (N, 3)
        T: (4, 4)

    Returns:
        (N, 3) transformed points.
    """
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    hom = torch.cat([points, ones], dim=-1)  # (N, 4)
    out = (T @ hom.T).T  # (N, 4)
    return out[:, :3]


def points_to_vehicle_frame(
    points: torch.Tensor,
    extrinsic: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """OpenCV camera frame → Waymo vehicle frame.

    Applies T_CV_TO_WAYMO then the camera extrinsic.
    """
    T_cv = _ensure_device(_T_CV_TO_WAYMO, device)
    points = transform_points(points, T_cv)
    points = transform_points(points, extrinsic)
    return points


def rasterize_bev_gpu(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    bev_range: float = BEV_RANGE,
    bev_res: float = BEV_RES,
    bev_size: int = BEV_SIZE,
) -> torch.Tensor:
    """Project (N,3) vehicle-frame points to a (4, H, W) BEV tensor on GPU.

    Channels:
        0 — max height
        1 — mean height
        2 — log point density
        3 — mean luminance
    """
    device = points.device
    N = points.shape[0]
    if N == 0:
        return torch.zeros(4, bev_size, bev_size, device=device)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Vehicle-frame → pixel coordinates
    px = ((bev_range - x) / bev_res).long()  # row  (forward axis)
    py = ((bev_range - y) / bev_res).long()  # col  (lateral axis)

    # Mask to valid grid cells
    mask = (px >= 0) & (px < bev_size) & (py >= 0) & (py < bev_size)
    px, py, z = px[mask], py[mask], z[mask]

    if colors is not None:
        lum = colors[mask].mean(dim=-1).float()
    else:
        lum = torch.zeros(px.shape[0], device=device)

    # Flatten 2D → 1D cell index for scatter
    cell = px * bev_size + py  # (M,)
    num_cells = bev_size * bev_size

    # ── Scatter-reduce per channel ──
    # Count
    count = torch.zeros(num_cells, device=device)
    count.scatter_add_(0, cell, torch.ones_like(cell, dtype=torch.float32))

    # Sum of Z (for mean)
    sum_z = torch.zeros(num_cells, device=device)
    sum_z.scatter_add_(0, cell, z)

    # Max Z
    max_z = torch.full((num_cells,), -1e6, device=device)
    max_z.scatter_reduce_(0, cell, z, reduce="amax", include_self=False)

    # Sum of luminance
    sum_lum = torch.zeros(num_cells, device=device)
    sum_lum.scatter_add_(0, cell, lum)

    # Assemble BEV
    occupied = count > 0
    bev = torch.zeros(4, num_cells, device=device)
    bev[0, occupied] = max_z[occupied]
    bev[1, occupied] = sum_z[occupied] / count[occupied]
    bev[2, occupied] = torch.log1p(count[occupied])
    bev[3, occupied] = sum_lum[occupied] / count[occupied]

    return bev.view(4, bev_size, bev_size)


# ── End-to-end pipeline ─────────────────────────────────────────────────────

def create_bev_from_frame_gpu(
    frame,
    depth_model,
    depth_processor,
    device: torch.device,
    max_depth: float = 50.0,
) -> np.ndarray:
    """End-to-end: Waymo proto frame → BEV array (4, 200, 200).

    All heavy geometry lives on GPU.  Only JPEG decode and undistortion use
    CPU (OpenCV), which is negligible.

    Returns:
        numpy array (4, BEV_SIZE, BEV_SIZE) float32 — ready to np.save().
    """

    # ── Extract calibrations ──
    calib_by_name = {}
    for calib in frame.frame.context.camera_calibrations:
        intr = list(calib.intrinsic)  # [fu, fv, cu, cv, k1, k2, p1, p2, k3]
        extr = np.array(list(calib.extrinsic.transform), dtype=np.float64).reshape(4, 4)
        calib_by_name[calib.name] = {
            "intrinsic": intr,
            "extrinsic": extr,
            "width": calib.width,
            "height": calib.height,
        }

    # ── Phase 1: Decode images & collect calibrations ──
    rgb_images = []
    intrinsics_list = []
    extrinsics_list = []

    for img_proto in frame.frame.images:
        cam_name = img_proto.name
        if cam_name not in calib_by_name:
            continue
        cal = calib_by_name[cam_name]

        # Decode JPEG (CPU — fast)
        jpg_bytes = np.frombuffer(img_proto.image, dtype=np.uint8)
        rgb = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        intr = cal["intrinsic"]
        intrinsics_list.append(
            [intr[0], intr[1], intr[2], intr[3],
             cal["width"], cal["height"]] + intr[4:9]
        )
        extrinsics_list.append(cal["extrinsic"])
        rgb_images.append(rgb)

    if not rgb_images:
        return np.zeros((4, BEV_SIZE, BEV_SIZE), dtype=np.float32)

    # ── Phase 2: Batched depth inference (GPU — unchanged) ──
    original_sizes = [(img.shape[0], img.shape[1]) for img in rgb_images]
    max_h = max(s[0] for s in original_sizes)
    max_w = max(s[1] for s in original_sizes)

    # Pad all images to uniform size for batching
    padded_images = []
    for img in rgb_images:
        h, w = img.shape[:2]
        padded = cv2.copyMakeBorder(
            img,
            top=0, bottom=max_h - h,
            left=0, right=max_w - w,
            borderType=cv2.BORDER_REPLICATE,
        )
        padded_images.append(padded)

    inputs = depth_processor(images=padded_images, return_tensors="pt").to(device)
    with torch.no_grad():
        preds = depth_model(**inputs).predicted_depth  # (N, pred_h, pred_w)

    # Interpolate each prediction back to original size and crop padding
    # Keep depth on GPU as float32-- no .cpu()!
    depth_maps_gpu = []
    for i, (orig_h, orig_w) in enumerate(original_sizes):
        pred_i = preds[i:i+1].unsqueeze(1).float()  # (1, 1, pred_h, pred_w)
        pred_i_full = F.interpolate(
            pred_i,
            size=(max_h, max_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()  # (max_h, max_w)

        # Crop to original camera resolution
        depth_cropped = pred_i_full[:orig_h, :orig_w]

        # Clamp to valid range (direct metric depth from model)
        depth_cropped = depth_cropped.clamp(0.1, max_depth)
        depth_maps_gpu.append(depth_cropped)

    # ── Phase 3: Per-camera back-project + transform (all on GPU) ──
    all_points = []
    all_colors = []

    for i, (rgb, depth_gpu, intr_vals, extr_np) in enumerate(
        zip(rgb_images, depth_maps_gpu, intrinsics_list, extrinsics_list)
    ):
        fx, fy, cx, cy = intr_vals[0], intr_vals[1], intr_vals[2], intr_vals[3]
        w, h = int(intr_vals[4]), int(intr_vals[5])

        # Undistort RGB on CPU if distortion coefficients provided
        if len(intr_vals) > 6:
            dist_coeffs = intr_vals[6:11]
            rgb = undistort_image(rgb, fx, fy, cx, cy, dist_coeffs)
            # Undistort depth: move to CPU briefly, undistort, back to GPU
            depth_np = depth_gpu.cpu().numpy()
            depth_np = undistort_image(depth_np, fx, fy, cx, cy, dist_coeffs)
            depth_gpu = torch.from_numpy(depth_np).to(device)

        # Back-project depth → 3D points in OpenCV camera frame
        pts = backproject_depth_to_points(depth_gpu, fx, fy, cx, cy)  # (H*W, 3)

        # Filter out invalid / too-close / too-far points
        d_flat = depth_gpu.reshape(-1)
        valid = (d_flat > 0.1) & (d_flat < max_depth)
        pts = pts[valid]

        # Get corresponding RGB colours (normalised to [0, 1])
        rgb_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).to(device)
        rgb_flat = rgb_tensor.reshape(-1, 3)
        colors = rgb_flat[valid]

        # Transform to vehicle frame
        extr_gpu = torch.from_numpy(extr_np.astype(np.float32)).to(device)
        pts = points_to_vehicle_frame(pts, extr_gpu, device)

        all_points.append(pts)
        all_colors.append(colors)

    # ── Fuse all cameras ──
    if not all_points:
        return np.zeros((4, BEV_SIZE, BEV_SIZE), dtype=np.float32)

    fused_pts = torch.cat(all_points, dim=0)    # (total_N, 3)
    fused_col = torch.cat(all_colors, dim=0)    # (total_N, 3)

    # Range-based filtering (replaces statistical outlier removal)
    # Keep only points within the BEV grid + reasonable height
    x, y, z = fused_pts[:, 0], fused_pts[:, 1], fused_pts[:, 2]
    keep = (
        (x.abs() < BEV_RANGE) &
        (y.abs() < BEV_RANGE) &
        (z > -3.0) &  # below road surface
        (z < 10.0)    # above tallest vehicles / structures
    )
    fused_pts = fused_pts[keep]
    fused_col = fused_col[keep]

    # ── Phase 4: BEV rasterisation (GPU) ──
    bev = rasterize_bev_gpu(fused_pts, fused_col)

    return bev.cpu().numpy()
