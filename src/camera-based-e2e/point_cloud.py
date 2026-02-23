import open3d as o3d
import numpy as np

import open3d as o3d
from protos import dataset_pb2 # Standard waymo import

def get_waymo_intrinsics(frame, camera_name_enum):
    """
    Extracts intrinsics for a specific camera from a Waymo Frame proto.
    
    Args:
        frame: The Waymo Open Dataset Frame proto.
        camera_name_enum: Integer enum for the camera (e.g., 1 for FRONT).
    
    Returns:
        o3d.camera.PinholeCameraIntrinsic object.
    """
    # 1. Find the calibration for the requested camera
    calib = None
    for c in frame.context.camera_calibrations:
        if c.name == camera_name_enum:
            calib = c
            break

    # Proto definition: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
    fx = calib.intrinsic[0]
    fy = calib.intrinsic[1]
    cx = calib.intrinsic[2]
    cy = calib.intrinsic[3]
    
    width = calib.width
    height = calib.height

    # 3. Create Open3D Intrinsic Object
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)
    
    return intrinsics

def create_point_cloud(rgb_image, depth_map, intrinsics=None, depth_scale=1.0):
    """
    Converts a single RGB image and Depth map into a 3D Point Cloud.
    
    Args:
        rgb_image (np.array): Shape (H, W, 3) - uint8 [0-255]
        depth_map (np.array): Shape (H, W) - float32
        intrinsics (o3d.camera.PinholeCameraIntrinsic): Optional custom intrinsics.
    """
    height, width = depth_map.shape

    # 1. Create Open3D Image objects
    o3d_color = o3d.geometry.Image(rgb_image)
    o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))

    # 2. Create RGBD Image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, 
        o3d_depth, 
        depth_scale=1.0, 
        depth_trunc=10000.0, 
        convert_rgb_to_intensity=False
    )

    # 4. Back-project to Point Cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        intrinsics
    )
    
    # Flip the point cloud because Open3D uses Y-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    return pcd
