import open3d as o3d
import numpy as np
import cv2

from protos import dataset_pb2  # Standard waymo import

# OpenCV camera frame (+X right, +Y down, +Z forward)
# -> Waymo camera frame (+X forward, +Y left, +Z up)
T_CV_TO_WAYMO = np.array([
    [ 0,  0,  1, 0],
    [-1,  0,  0, 0],
    [ 0, -1,  0, 0],
    [ 0,  0,  0, 1],
], dtype=np.float64)

# Waymo vehicle frame (+X forward, +Y left, +Z up)
# -> Open3D visualization frame (+X right, +Y up, -Z forward/backward)
T_WAYMO_TO_O3D = np.array([
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1],
], dtype=np.float64)


def disparity_to_depth(disparity, max_depth=50.0):
    # Monocular model has disparity (inverse depth), so reverting that and normalizing it
    disp_normalized = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)
    true_depth = 1.0 / (disp_normalized + 0.01)
    true_depth = np.clip(true_depth, 0.1, max_depth)
    return true_depth.astype(np.float32)


def compute_metric_scale(pcd, extrinsic, ground_percentile=10):
    # compute a sale factor by setting the ground plane to Z = 0
    # We know the actual height of the camera above ground from the extrinsic matrix
    
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return 1.0

    cam_height = extrinsic[2, 3]          # true height of camera above ground (m)
    if abs(cam_height) < 0.1:             # no valid height info (e.g. identity extrinsic)
        return 1.0

    z_vals = pts[:, 2]                    # Z vals in pcd
    z_thresh = np.percentile(z_vals, ground_percentile) # gets the threshold for 10th percentile
    ground_mask = z_vals <= z_thresh
    z_ground_raw = np.median(z_vals[ground_mask]) #identify median of 10th percentile pts and sets that as our ground

    # compute the scale factor
    denom = cam_height - z_ground_raw
    if abs(denom) < 1e-6:
        return 1.0

    scale = cam_height / denom
    return float(scale)


def undistort_image(image, fx, fy, cx, cy, dist_coeffs):
    # Removes lens distortion from an image using OpenCV.
    # dist_coeffs: [k1, k2, p1, p2, k3] — same order as OpenCV and Waymo proto
    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    return cv2.undistort(image, camera_matrix, dist)


def create_point_cloud(rgb_image, depth_map, intrinsics, depth_scale=1.0):
    #converts a single rgb image into a point cloud

    # Create Open3D Image objects
    o3d_color = o3d.geometry.Image(rgb_image)
    o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))

    # Create RGBD Image, rgb + depth
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale, 
        depth_trunc=10000.0,
        convert_rgb_to_intensity=False #make sure we have color images
    )

    # This is the actual back-projection to a point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    return pcd


def transform_to_vehicle_frame(pcd, extrinsic):
    # Transforms a point cloud from OpenCV camera frame to the Waymo vehicle frame.

    # OpenCV (+X right, +Y down, +Z forward) -> Waymo (+X forward, +Y left, +Z up)
    pcd.transform(T_CV_TO_WAYMO)

    # Waymo camera frame -> vehicle frame
    # use the 4x4 matrix in camera extrinsics
    pcd.transform(extrinsic)

    return pcd


def merge_point_clouds(point_clouds):
    # Literally just adds point clouds together

    merged = o3d.geometry.PointCloud()

    for pcd in point_clouds:
        merged += pcd

    return merged


def create_multi_view_point_cloud(rgb_images, depth_maps, intrinsics_list, extrinsics_list, depth_scale=1.0):
    # Fuses point clouds from multiple cameras into one in vehicle frame.
    # rgb_images:       list of (H, W, 3) uint8 arrays
    # depth_maps:       list of (H, W) float32 arrays
    # intrinsics_list:  list of [fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3] arrays
    # extrinsics_list:  list of (4, 4) arrays (camera -> vehicle)

    per_camera_clouds = []

    for rgb, depth, intr_vals, extr in zip(rgb_images, depth_maps, intrinsics_list, extrinsics_list):
        fx, fy, cx, cy = intr_vals[0], intr_vals[1], intr_vals[2], intr_vals[3]
        w, h = int(intr_vals[4]), int(intr_vals[5])

        # Undistort RGB and depth if distortion coefficients are provided
        if len(intr_vals) > 6:
            dist_coeffs = intr_vals[6:11]  # [k1, k2, p1, p2, k3]
            rgb = undistort_image(rgb, fx, fy, cx, cy, dist_coeffs)
            depth = undistort_image(depth, fx, fy, cx, cy, dist_coeffs)

        # Build Open3D intrinsics (pinhole, no distortion — image is already undistorted)
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            width=w, height=h,
            fx=fx, fy=fy,
            cx=cx, cy=cy
        )

        # Invert disparity to true depth
        depth = disparity_to_depth(depth)

        # Back-project to point cloud (OpenCV frame)
        pcd = create_point_cloud(rgb, depth, intrinsics, depth_scale=depth_scale)

        # OpenCV -> Waymo camera -> vehicle frame
        transform_to_vehicle_frame(pcd, extr)

        # ── Per-camera metric scale correction ──
        # Scale the cloud about the camera origin so the ground plane lands at Z=0
        scale = compute_metric_scale(pcd, extr)
        cam_origin = extr[:3, 3]  # camera position in vehicle frame
        pts = np.asarray(pcd.points)
        pts = (pts - cam_origin) * scale + cam_origin
        pcd.points = o3d.utility.Vector3dVector(pts)

        per_camera_clouds.append(pcd)

    fused_pcd = merge_point_clouds(per_camera_clouds)

    # Remove statistical outliers (stray/noisy points)
    if len(fused_pcd.points) > 0:
        fused_pcd, _ = fused_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

    # Finally, rotate vehicle frame into Open3D visualization frame
    fused_pcd.transform(T_WAYMO_TO_O3D)

    return fused_pcd


# ── BEV Rasterization ──────────────────────────────────────────────────

# Default BEV grid parameters
BEV_RANGE = 25.0     # meters in each direction from ego
BEV_RES   = 0.25     # meters per pixel
BEV_SIZE  = int(2 * BEV_RANGE / BEV_RES)  # 200


def rasterize_bev(
    pcd,
    bev_range: float = BEV_RANGE,
    bev_res: float = BEV_RES,
    bev_size: int = BEV_SIZE,
):
    """Project a point cloud (in vehicle frame) onto a top-down BEV grid.

    The point cloud should be in the **Waymo vehicle frame** (+X forward,
    +Y left, +Z up) — i.e. call this *before* applying T_WAYMO_TO_O3D.

    Returns:
        bev: np.ndarray of shape (4, bev_size, bev_size) float32
             channel 0 – max height
             channel 1 – mean height
             channel 2 – log point density
             channel 3 – mean luminance (from colour if available, else 0)
    """
    pts = np.asarray(pcd.points)  # (N, 3) — X fwd, Y left, Z up
    has_color = pcd.has_colors()
    colors = np.asarray(pcd.colors) if has_color else None  # (N, 3) in [0,1]

    bev = np.zeros((4, bev_size, bev_size), dtype=np.float32)
    if len(pts) == 0:
        return bev

    # Map vehicle-frame (X fwd, Y left) → pixel coords
    # Pixel (0,0) = top-left = (max_X, max_Y) of scene
    # row ↔ forward axis (X), col ↔ lateral axis (Y)
    px = ((bev_range - pts[:, 0]) / bev_res).astype(np.int32)  # row
    py = ((bev_range - pts[:, 1]) / bev_res).astype(np.int32)  # col

    # Clip to grid
    mask = (px >= 0) & (px < bev_size) & (py >= 0) & (py < bev_size)
    px, py, z = px[mask], py[mask], pts[mask, 2]
    if colors is not None:
        lum = colors[mask].mean(axis=1).astype(np.float32)
    else:
        lum = np.zeros(len(px), dtype=np.float32)

    # Accumulate per-cell statistics
    count = np.zeros((bev_size, bev_size), dtype=np.float32)
    sum_z = np.zeros((bev_size, bev_size), dtype=np.float32)
    max_z = np.full((bev_size, bev_size), -1e6, dtype=np.float32)
    sum_lum = np.zeros((bev_size, bev_size), dtype=np.float32)

    np.add.at(count, (px, py), 1)
    np.add.at(sum_z, (px, py), z)
    np.maximum.at(max_z, (px, py), z)
    np.add.at(sum_lum, (px, py), lum)

    occupied = count > 0
    bev[0][occupied] = max_z[occupied]
    bev[1][occupied] = sum_z[occupied] / count[occupied]
    bev[2][occupied] = np.log1p(count[occupied])
    bev[3][occupied] = sum_lum[occupied] / count[occupied]

    return bev


def create_bev_from_frame(frame, depth_model, depth_processor, device, max_depth=50.0):
    """End-to-end: E2E proto frame → BEV array (4, 200, 200).

    Parameters
    ----------
    frame : e2e_pb2.E2EDFrame (already parsed)
    depth_model : HuggingFace depth model (e.g. Depth-Anything-V2)
    depth_processor : corresponding AutoImageProcessor
    device : torch device for depth inference
    max_depth : clip depth beyond this range (meters)
    """
    import torch
    import torch.nn.functional as F

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

    # ── Phase 1: Decode all images & collect calibrations ──
    rgb_images = []
    intrinsics_list = []
    extrinsics_list = []

    for img_proto in frame.frame.images:
        cam_name = img_proto.name
        if cam_name not in calib_by_name:
            continue
        cal = calib_by_name[cam_name]

        # Decode JPEG
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

    # ── Phase 2: Batched depth inference (all cameras in one forward pass) ──
    original_sizes = [(img.shape[0], img.shape[1]) for img in rgb_images]
    max_h = max(s[0] for s in original_sizes)
    max_w = max(s[1] for s in original_sizes)

    # Pad all images to (max_h, max_w) using edge replication
    # so they can all be batched together WITHOUT aspect-ratio squishing.
    padded_images = []
    for img in rgb_images:
        h, w = img.shape[:2]
        padded = cv2.copyMakeBorder(
            img, 
            top=0, bottom=max_h - h, 
            left=0, right=max_w - w, 
            borderType=cv2.BORDER_REPLICATE
        )
        padded_images.append(padded)

    inputs = depth_processor(images=padded_images, return_tensors="pt").to(device)
    with torch.no_grad():
        preds = depth_model(**inputs).predicted_depth  # (N, pred_h, pred_w)

    # Interpolate each prediction back to the padded size, then crop to original size
    depth_maps = []
    for i, (orig_h, orig_w) in enumerate(original_sizes):
        pred_i = preds[i : i + 1].unsqueeze(1)  # (1, 1, pred_h, pred_w)
        pred_i_full = F.interpolate(
            pred_i,
            size=(max_h, max_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        # Crop the padded region out
        pred_i_cropped = pred_i_full[:orig_h, :orig_w]
        depth_maps.append(disparity_to_depth(pred_i_cropped, max_depth=max_depth))

    # ── Phase 3: Build fused point cloud (vehicle frame) ──
    per_camera_clouds = []
    for rgb, depth, intr_vals, extr in zip(rgb_images, depth_maps, intrinsics_list, extrinsics_list):
        fx, fy, cx, cy = intr_vals[0], intr_vals[1], intr_vals[2], intr_vals[3]
        w, h = int(intr_vals[4]), int(intr_vals[5])

        if len(intr_vals) > 6:
            dist_coeffs = intr_vals[6:11]
            rgb = undistort_image(rgb, fx, fy, cx, cy, dist_coeffs)
            depth = undistort_image(depth, fx, fy, cx, cy, dist_coeffs)

        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

        pcd = create_point_cloud(rgb, depth, intrinsics)
        transform_to_vehicle_frame(pcd, extr)

        scale = compute_metric_scale(pcd, extr)
        cam_origin = extr[:3, 3]
        pts = np.asarray(pcd.points)
        pts = (pts - cam_origin) * scale + cam_origin
        pcd.points = o3d.utility.Vector3dVector(pts)

        per_camera_clouds.append(pcd)

    fused_pcd = merge_point_clouds(per_camera_clouds)
    if len(fused_pcd.points) > 0:
        fused_pcd, _ = fused_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # NOTE: do NOT apply T_WAYMO_TO_O3D — rasterize in vehicle frame
    return rasterize_bev(fused_pcd)
