"""Offline occupancy pseudo-label generation.
For each frame in the Waymo E2E dataset this script:
  1. Decodes all camera images.
  2. Runs SegFormer-B2 to produce per-pixel semantic masks.
  3. Runs Depth-Anything-V2 to get metric depth maps.
  4. Back-projects depth -> 3D points in vehicle frame.
  5. Assigns semantic class to each point from the segmentation mask.
  6. Filters points that appear in fewer than min_views cameras (multiview filter).
  7. Ray-casts from each camera origin to mark free space (numba JIT).
  8. Voxelises into a (100, 100, 16) uint8 occupancy grid with majority voting.

Semantic class map (Cityscapes → our 6 classes):
  0  free       (confirmed empty along camera rays)
  1  vehicle
  2  pedestrian
  3  cyclist
  4  road
  5  static
  255 unknown   (unobserved — ignored in loss)
"""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numba
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerForSemanticSegmentation,
)

from protos import e2e_pb2
from point_cloud_gpu import (
    BEV_RANGE,
    backproject_depth_to_points,
    points_to_vehicle_frame,
    undistort_image,
)

# voxel grid config
VOX_XY_RANGE = 25.0
VOX_XY_RES   = 0.5
VOX_Z_MIN    = -3.0
VOX_Z_MAX    = 5.0
VOX_Z_RES    = 0.5
VOX_XY_SIZE  = int(2 * VOX_XY_RANGE / VOX_XY_RES)  # 100
VOX_Z_SIZE   = int((VOX_Z_MAX - VOX_Z_MIN) / VOX_Z_RES)  # 16
NUM_CLASSES  = 6

RAY_SUBSAMPLE = 8

# cityscapes -> our map
CITYSCAPES_TO_OCC = np.full(256, 5, dtype=np.uint8)
CITYSCAPES_TO_OCC[0]   = 4    # road
CITYSCAPES_TO_OCC[1]   = 5    # sidewalk
CITYSCAPES_TO_OCC[2]   = 5    # building
CITYSCAPES_TO_OCC[3]   = 5    # wall
CITYSCAPES_TO_OCC[4]   = 5    # fence
CITYSCAPES_TO_OCC[5]   = 5    # pole
CITYSCAPES_TO_OCC[6]   = 5    # traffic light
CITYSCAPES_TO_OCC[7]   = 5    # traffic sign
CITYSCAPES_TO_OCC[8]   = 5    # vegetation
CITYSCAPES_TO_OCC[9]   = 5    # terrain
CITYSCAPES_TO_OCC[10]  = 255  # sky
CITYSCAPES_TO_OCC[11]  = 2    # person
CITYSCAPES_TO_OCC[12]  = 3    # rider
CITYSCAPES_TO_OCC[13]  = 1    # car
CITYSCAPES_TO_OCC[14]  = 1    # truck
CITYSCAPES_TO_OCC[15]  = 1    # bus
CITYSCAPES_TO_OCC[16]  = 1    # train
CITYSCAPES_TO_OCC[17]  = 3    # motorcycle
CITYSCAPES_TO_OCC[18]  = 3    # bicycle
CITYSCAPES_TO_OCC[255] = 0    # ignore → free

# model loading

def load_seg_model(model_id: str, device: torch.device):
    print(f"Loading segmentation model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    if device.type == "cuda":
        model = model.half()
    return model, processor


def load_depth_model(device: torch.device):
    depth_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading depth model: {depth_id}")
    processor = AutoImageProcessor.from_pretrained(depth_id)
    model = AutoModelForDepthEstimation.from_pretrained(depth_id).to(device)
    model.eval()
    if device.type == "cuda":
        model = model.half()
    return model, processor

# segmentation

def run_segmentation_batch(rgb_images, seg_model, seg_processor, device):
    inputs = seg_processor(images=rgb_images, return_tensors="pt")
    inputs = {
        k: v.half().to(device) if v.dtype == torch.float32 else v.to(device)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    masks = []
    for i, img in enumerate(rgb_images):
        orig_h, orig_w = img.shape[:2]
        pred = logits[i:i+1].float()
        pred_up = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        class_map = pred_up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        masks.append(CITYSCAPES_TO_OCC[class_map])
    return masks

# depth inference

def run_depth_batch(rgb_images, depth_model, depth_processor, device, max_depth=30.0):
    """Handles cameras with different resolutions by padding to a common size."""
    original_sizes = [(img.shape[0], img.shape[1]) for img in rgb_images]
    max_h = max(s[0] for s in original_sizes)
    max_w = max(s[1] for s in original_sizes)
    padded = []
    for img in rgb_images:
        h, w = img.shape[:2]
        p = cv2.copyMakeBorder(img, 0, max_h - h, 0, max_w - w, cv2.BORDER_REPLICATE)
        padded.append(p)
    inputs = depth_processor(images=padded, return_tensors="pt").to(device)
    with torch.no_grad():
        preds = depth_model(**inputs).predicted_depth
    depth_maps = []
    for i, (orig_h, orig_w) in enumerate(original_sizes):
        pred_i = preds[i:i+1].unsqueeze(1).float()
        pred_full = F.interpolate(
            pred_i, size=(max_h, max_w), mode="bicubic", align_corners=False
        ).squeeze()
        depth_maps.append(pred_full[:orig_h, :orig_w].clamp(0.1, max_depth))
    return depth_maps

# multiview filter (vectorized)

def multiview_filter(all_pts, all_labels, min_views: int = 2):
    """Keep only points whose voxel is observed by at least min_views cameras.

    Uses fully vectorized numpy ops — no Python loops over points.
    Points from dynamic classes (vehicle/ped/cyclist) skip the filter
    since they're unlikely to overlap across cameras anyway.
    """
    num_vox = VOX_XY_SIZE * VOX_XY_SIZE * VOX_Z_SIZE

    # compute valid voxel flat index for every point from every camera
    cam_vox_keys = []   # flat index per point (-1 = out of bounds)
    cam_valid_masks = []  # which points are in-bounds
    for pts in all_pts:
        pts_np = pts.cpu().numpy()
        ix = np.floor((VOX_XY_RANGE - pts_np[:, 0]) / VOX_XY_RES).astype(np.int32)
        iy = np.floor((VOX_XY_RANGE - pts_np[:, 1]) / VOX_XY_RES).astype(np.int32)
        iz = np.floor((pts_np[:, 2] - VOX_Z_MIN) / VOX_Z_RES).astype(np.int32)
        in_bounds = (
            (ix >= 0) & (ix < VOX_XY_SIZE) &
            (iy >= 0) & (iy < VOX_XY_SIZE) &
            (iz >= 0) & (iz < VOX_Z_SIZE)
        )
        flat = np.where(in_bounds,
                        ix * VOX_XY_SIZE * VOX_Z_SIZE + iy * VOX_Z_SIZE + iz,
                        -1)
        cam_vox_keys.append(flat)
        cam_valid_masks.append(in_bounds)

    # count how many distinct cameras observe each voxel
    view_counts = np.zeros(num_vox, dtype=np.int8)
    for flat in cam_vox_keys:
        valid_flat = flat[flat >= 0]
        unique_vox = np.unique(valid_flat)
        view_counts[unique_vox] += 1

    # Fflter each camera's points — dynamic objects bypass the view count check
    filtered_pts, filtered_labels = [], []
    for pts, labels, flat, in_bounds in zip(all_pts, all_labels, cam_vox_keys, cam_valid_masks):
        lbl_np = labels.cpu().numpy()
        is_dynamic = (lbl_np >= 1) & (lbl_np <= 3)
        # out-of-bounds points get view_count=0, so they're dropped for static
        seen_enough = np.zeros(len(flat), dtype=bool)
        valid_idx = np.where(in_bounds)[0]
        seen_enough[valid_idx] = view_counts[flat[valid_idx]] >= min_views
        keep = is_dynamic | seen_enough
        filtered_pts.append(pts[torch.from_numpy(keep).to(pts.device)])
        filtered_labels.append(labels[torch.from_numpy(keep).to(labels.device)])

    return filtered_pts, filtered_labels

# DDA ray casting - marks intermediate voxels as free, modifies voxels marked unknown

@numba.njit
def _dda_ray(ox: float, oy: float, oz: float, ex: float, ey: float, ez: float, grid: np.ndarray) -> None:
    dx = ex - ox
    dy = ey - oy
    dz = ez - oz
    n = int(max(abs(dx), abs(dy), abs(dz)))
    if n < 2 or n > 90:  # 90 voxels = 45m
        return
    step_x = dx / n
    step_y = dy / n
    step_z = dz / n
    cx = float(ox)
    cy = float(oy)
    cz = float(oz)
    for _ in range(n - 1):
        cx += step_x
        cy += step_y
        cz += step_z
        vx = int(cx)
        vy = int(cy)
        vz = int(cz)
        if 0 <= vx < 100 and 0 <= vy < 100 and 0 <= vz < 16:
            if grid[vx, vy, vz] == 255:
                grid[vx, vy, vz] = 0


def mark_free_space(grid: np.ndarray, cam_origin: np.ndarray,
                    pts_veh: np.ndarray) -> np.ndarray:
    if len(pts_veh) == 0:
        return grid
    ox = (VOX_XY_RANGE - cam_origin[0]) / VOX_XY_RES
    oy = (VOX_XY_RANGE - cam_origin[1]) / VOX_XY_RES
    oz = (cam_origin[2] - VOX_Z_MIN) / VOX_Z_RES
    pts_sub = pts_veh[::RAY_SUBSAMPLE]
    px = (VOX_XY_RANGE - pts_sub[:, 0]) / VOX_XY_RES
    py = (VOX_XY_RANGE - pts_sub[:, 1]) / VOX_XY_RES
    pz = (pts_sub[:, 2] - VOX_Z_MIN) / VOX_Z_RES
    for i in range(len(px)):
        _dda_ray(ox, oy, oz, px[i], py[i], pz[i], grid)
    return grid

# voxelisation with majority voting

def voxelise_labelled_cloud(pts: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    x   = pts[:, 0].cpu().numpy()
    y   = pts[:, 1].cpu().numpy()
    z   = pts[:, 2].cpu().numpy()
    lbl = labels.cpu().numpy()
    valid_lbl = lbl < NUM_CLASSES
    x, y, z, lbl = x[valid_lbl], y[valid_lbl], z[valid_lbl], lbl[valid_lbl]
    ix = ((VOX_XY_RANGE - x) / VOX_XY_RES).astype(np.int32)
    iy = ((VOX_XY_RANGE - y) / VOX_XY_RES).astype(np.int32)
    iz = ((z - VOX_Z_MIN) / VOX_Z_RES).astype(np.int32)
    valid = (
        (ix >= 0) & (ix < VOX_XY_SIZE) &
        (iy >= 0) & (iy < VOX_XY_SIZE) &
        (iz >= 0) & (iz < VOX_Z_SIZE)
    )
    ix, iy, iz, lbl = ix[valid], iy[valid], iz[valid], lbl[valid]
    flat_idx = ix * VOX_XY_SIZE * VOX_Z_SIZE + iy * VOX_Z_SIZE + iz
    num_vox  = VOX_XY_SIZE * VOX_XY_SIZE * VOX_Z_SIZE
    counts   = np.zeros((num_vox, NUM_CLASSES), dtype=np.int32)
    np.add.at(counts, (flat_idx, lbl.astype(np.int32)), 1)
    counts_sum = counts.sum(axis=1)
    winner     = counts.argmax(axis=1).astype(np.uint8)
    occupied   = counts_sum > 0

    is_static_road = np.isin(winner, [4, 5])
    strong = counts_sum >= 1
    keep   = occupied & (~is_static_road | strong)

    flat_grid = np.full(num_vox, 255, dtype=np.uint8)
    flat_grid[keep] = winner[keep]
    return flat_grid.reshape(VOX_XY_SIZE, VOX_XY_SIZE, VOX_Z_SIZE)

# per-frame processing

def process_frame(frame, seg_model, seg_processor, depth_model, depth_processor,
                  device, max_depth=30.0, min_views=1):
    """Full pipeline for one frame -> (100, 100, 16) uint8 occupancy grid."""

    calib_by_name = {}
    for calib in frame.frame.context.camera_calibrations:
        intr = list(calib.intrinsic)
        extr = np.array(list(calib.extrinsic.transform), dtype=np.float64).reshape(4, 4)
        calib_by_name[calib.name] = {
            "intrinsic": intr,
            "extrinsic": extr,
            "width":     calib.width,
            "height":    calib.height,
        }

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
        intrinsics_list.append(
            [intr[0], intr[1], intr[2], intr[3], cal["width"], cal["height"]] + intr[4:9]
        )
        extrinsics_list.append(cal["extrinsic"])
        rgb_images.append(rgb)

    if not rgb_images:
        return np.full((VOX_XY_SIZE, VOX_XY_SIZE, VOX_Z_SIZE), 255, dtype=np.uint8)

    seg_maps       = run_segmentation_batch(rgb_images, seg_model, seg_processor, device)
    depth_maps_gpu = run_depth_batch(rgb_images, depth_model, depth_processor, device, max_depth)

    all_pts, all_labels, cam_origins = [], [], []

    for rgb, depth_gpu, intr_vals, extr_np, seg_mask in zip(
        rgb_images, depth_maps_gpu, intrinsics_list, extrinsics_list, seg_maps
    ):
        fx, fy, cx, cy = intr_vals[0], intr_vals[1], intr_vals[2], intr_vals[3]

        if len(intr_vals) > 6:
            dist = intr_vals[6:11]
            rgb       = undistort_image(rgb, fx, fy, cx, cy, dist)
            depth_np  = depth_gpu.cpu().numpy()
            depth_np  = undistort_image(depth_np, fx, fy, cx, cy, dist)
            depth_gpu = torch.from_numpy(depth_np).to(device)
            seg_mask  = undistort_image(seg_mask, fx, fy, cx, cy, dist)

        pts_cv = backproject_depth_to_points(depth_gpu, fx, fy, cx, cy)
        d_flat = depth_gpu.reshape(-1)
        valid  = (d_flat > 0.1) & (d_flat < max_depth)
        pts_cv = pts_cv[valid]

        extr_gpu = torch.from_numpy(extr_np.astype(np.float32)).to(device)
        pts_veh  = points_to_vehicle_frame(pts_cv, extr_gpu, device)

        x_, y_, z_ = pts_veh[:, 0], pts_veh[:, 1], pts_veh[:, 2]
        keep = (
            (x_.abs() < VOX_XY_RANGE) & (y_.abs() < VOX_XY_RANGE) &
            (z_ > VOX_Z_MIN - 0.5) & (z_ < VOX_Z_MAX + 0.5)
        )
        pts_veh = pts_veh[keep]

        valid_idx  = valid.nonzero(as_tuple=True)[0]
        keep_idx   = keep.nonzero(as_tuple=True)[0]
        depth_vals = d_flat[valid_idx][keep_idx]

        H, W = depth_gpu.shape
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        u_flat = u_grid.reshape(-1)[valid_idx][keep_idx].long().clamp(0, W - 1)
        v_flat = v_grid.reshape(-1)[valid_idx][keep_idx].long().clamp(0, H - 1)
        seg_t  = torch.from_numpy(seg_mask.astype(np.int64)).to(device)
        point_labels = seg_t[v_flat, u_flat].to(torch.uint8)

        # remove sky points
        sky_mask = point_labels != 255
        pts_veh      = pts_veh[sky_mask]
        point_labels = point_labels[sky_mask]
        depth_vals   = depth_vals[sky_mask]

        # class-aware depth filter
        is_dynamic = (point_labels >= 1) & (point_labels <= 3)
        # depth_keep = (is_dynamic | (depth_vals < 15.0)) & (depth_vals < 25.0)
        depth_keep = depth_vals < 25.0
        pts_veh      = pts_veh[depth_keep]
        # print("raw pts:", pts_veh.shape[0])
        point_labels = point_labels[depth_keep]

        all_pts.append(pts_veh)
        all_labels.append(point_labels)
        cam_origins.append(extr_np[:3, 3])

    if not all_pts:
        return np.full((VOX_XY_SIZE, VOX_XY_SIZE, VOX_Z_SIZE), 255, dtype=np.uint8)

    # multiview filter — remove static noise seen by only one camera
    all_pts, all_labels = multiview_filter(all_pts, all_labels, min_views=min_views)

    fused_pts    = torch.cat(all_pts, dim=0)
    fused_labels = torch.cat(all_labels, dim=0)

    # ray cast on blank gris -> free space
    # Only use static + road points for ray casting
    grid = np.full((VOX_XY_SIZE, VOX_XY_SIZE, VOX_Z_SIZE), 255, dtype=np.uint8)
    for i, cam_origin in enumerate(cam_origins):
        lbl_np = all_labels[i].cpu().numpy()
        static_mask = lbl_np >= 4  # road (4) and static (5) only
        ray_pts = all_pts[i][torch.from_numpy(static_mask).to(all_pts[i].device)]
        grid = mark_free_space(grid, cam_origin, ray_pts.cpu().numpy())

    # overlay semantic labels with majority voting
    semantic_grid = voxelise_labelled_cloud(fused_pts, fused_labels)
    occupied_mask = semantic_grid != 255
    grid[occupied_mask] = semantic_grid[occupied_mask]
    return grid

def main():
    parser = argparse.ArgumentParser(description="Generate occupancy pseudo-labels")
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--split",       type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--index_dir",   type=str, default=".")
    parser.add_argument("--start_idx",   type=int, default=0)
    parser.add_argument("--n_items",     type=int, default=None)
    parser.add_argument("--seg_model",   type=str,
                        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    parser.add_argument("--min_views",   type=int, default=2,
                        help="Min cameras that must observe a static voxel to keep it")
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir",  type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    occ_base      = Path(args.output_dir) if args.output_dir else Path(args.data_dir).parent / "occ"
    occ_split_dir = occ_base / args.split
    occ_split_dir.mkdir(parents=True, exist_ok=True)

    index_file = os.path.join(args.index_dir, f"index_{args.split}.pkl")
    with open(index_file, "rb") as f:
        indexes = pickle.load(f)

    end_idx = len(indexes)
    if args.n_items is not None:
        end_idx = min(args.start_idx + args.n_items, end_idx)
    print(f"Processing frames {args.start_idx}–{end_idx - 1}")

    seg_model,   seg_processor   = load_seg_model(args.seg_model, device)
    depth_model, depth_processor = load_depth_model(device)

    _dummy = np.full((100, 100, 16), 255, dtype=np.uint8)
    _dda_ray(50.0, 50.0, 6.0, 70.0, 50.0, 6.0, _dummy)
    print("Numba JIT warmed up.")

    occ_index = []
    open_file, open_filename = None, ""

    for idx in tqdm(range(args.start_idx, end_idx), desc=f"OccLabels [{args.split}]"):
        occ_path = occ_split_dir / f"occ_{idx:07d}.npy"

        if occ_path.exists():
            occ_index.append((idx, str(occ_path)))
            continue

        filename, start_byte, byte_length = indexes[idx]
        if open_filename != filename:
            if open_file is not None:
                open_file.close()
            open_file     = open(os.path.join(args.data_dir, filename), "rb")
            open_filename = filename

        open_file.seek(start_byte)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(open_file.read(byte_length))

        occ_grid = process_frame(
            frame, seg_model, seg_processor, depth_model, depth_processor,
            device, min_views=args.min_views
        )

        np.save(str(occ_path), occ_grid)
        occ_index.append((idx, str(occ_path)))

    if open_file is not None:
        open_file.close()

    index_path = occ_base / f"occ_index_{args.split}.pkl"
    with open(str(index_path), "wb") as f:
        pickle.dump(occ_index, f)

    print(f"\nSaved {len(occ_index)} occupancy grids to {occ_split_dir}")
    print(f"Index written to {index_path}")


if __name__ == "__main__":
    main()

