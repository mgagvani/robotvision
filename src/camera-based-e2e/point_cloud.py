import open3d as o3d
import numpy as np
import cv2
from protos import dataset_pb2

T_CV_TO_WAYMO = np.array([
    [ 0,  0,  1, 0],
    [-1,  0,  0, 0],
    [ 0, -1,  0, 0],
    [ 0,  0,  0, 1],
], dtype=np.float64)

T_WAYMO_TO_O3D = np.array([
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1],
], dtype=np.float64)


def disparity_to_depth(disparity, max_depth=50.0):
    disp_normalized = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)
    true_depth = 1.0 / (disp_normalized + 0.01)
    true_depth = np.clip(true_depth, 0.1, max_depth)
    return true_depth.astype(np.float32)


def compute_metric_scale(pcd, extrinsic, ground_percentile=10):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return 1.0

    cam_height = extrinsic[2, 3]
    if abs(cam_height) < 0.1:
        return 1.0

    z_vals = pts[:, 2]
    z_thresh = np.percentile(z_vals, ground_percentile)
    ground_mask = z_vals <= z_thresh
    z_ground_raw = np.median(z_vals[ground_mask])

    denom = cam_height - z_ground_raw
    if abs(denom) < 1e-6:
        return 1.0

    scale = cam_height / denom
    return float(scale)


def undistort_image(image, fx, fy, cx, cy, dist_coeffs):
    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    return cv2.undistort(image, camera_matrix, dist)


def create_point_cloud(rgb_image, depth_map, intrinsics, depth_scale=1.0):
    o3d_color = o3d.geometry.Image(rgb_image)
    o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=10000.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    return pcd


def transform_to_vehicle_frame(pcd, extrinsic):
    pcd.transform(T_CV_TO_WAYMO)
    # pcd.transform(np.linalg.inv(extrinsic))
    pcd.transform(extrinsic)
    return pcd


def merge_point_clouds(point_clouds):
    merged = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        merged += pcd
    return merged


def create_multi_view_point_cloud(rgb_images, depth_maps, intrinsics_list, extrinsics_list, depth_scale=1.0):
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

        depth = disparity_to_depth(depth)

        pcd = create_point_cloud(rgb, depth, intrinsics, depth_scale=depth_scale)

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

    fused_pcd.transform(T_WAYMO_TO_O3D)

    return fused_pcd
