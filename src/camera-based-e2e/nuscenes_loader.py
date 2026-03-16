## Adapted from: https://github.com/mgagvani/e2e-driving-project/blob/main/src/nuscenes/nuscenes_dataset.py
import os
from functools import cache
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

GET_IMG_DATA = True #  set to false for ad_mlp

# helpers for debugging arbitrary python objects
def get_type_tree(data):
    """
    Recursively builds a tree of types from the input data, which can be a
    nested collection (dict, set, list, tuple, etc.).
    """
    if isinstance(data, (dict, list, tuple, set)):
        tree = {}
        if isinstance(data, dict):
            for key, value in data.items():
                tree[key] = get_type_tree(value)
        else:
            for idx, item in enumerate(data):
                tree[idx] = get_type_tree(item)
        return tree
    else:
        return type(data).__name__


def print_type_tree(tree, level=0):
    """
    Recursively prints the type tree with indentation to represent its structure.
    """
    indent = "  " * level
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}{key}:")
            print_type_tree(value, level + 1)
    else:
        print(f"{indent}{tree}")


def locate_message(utimes, utime):
    """Find the closest message by timestamp."""
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

def conditional_decorator(decorator: Callable, condition: bool) -> Callable:
    """
    Conditionally apply a decorator based on a boolean condition.

    Args:
        decorator (Callable): The decorator to apply.
        condition (bool): If True, apply the decorator; otherwise, return the original function.

    Returns:
        Callable: The decorated function if condition is True, else the original function.
    """
    def wrapper(func):
        if condition:
            return decorator(func)
        return func
    return wrapper


class NuScenesDataset(Dataset):
    def __init__(
        self, nuscenes_path, version="v1.0-trainval", future_seconds=3.0, future_hz=2, past_frames=4, split="train", get_img_data=True
    ):
        """
        Initialize the NuScenes dataset

        Args:
            nuscenes_path: Path to the NuScenes dataset root
            version: NuScenes dataset version
            future_seconds: Number of seconds in the future for trajectory prediction
            future_hz: Frequency in Hz for trajectory points
            past_frames: Number of past frames to collect for ego states (Tp=4)
            split: One of 'train', 'val', 'test', 'mini_train', 'mini_val'
        """
        self.nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
        self.nusc_can = NuScenesCanBus(dataroot=nuscenes_path)
        self.version = version
        self.split = split
        self.future_seconds = future_seconds
        self.future_hz = future_hz
        self.future_steps = int(future_seconds * future_hz)
        self.past_frames = past_frames
        global GET_IMG_DATA
        GET_IMG_DATA = get_img_data
        self.get_img_data = GET_IMG_DATA
        
        # Filter samples by split
        self.scenes = self._get_scenes()
        self.samples = self._filter_samples()

    def _get_scenes(self):
        """Get scene names for the specified split."""
        # Map version and split to the correct split name
        split_mapping = {
            'v1.0-trainval': {'train': 'train', 'val': 'val', 'test': 'test'},
            'v1.0-mini': {'train': 'mini_train', 'val': 'mini_val', 'mini_train': 'mini_train', 'mini_val': 'mini_val'},
        }
        
        if self.version not in split_mapping:
            raise ValueError(f"Unsupported version: {self.version}")
        
        if self.split not in split_mapping[self.version]:
            raise ValueError(f"Unsupported split '{self.split}' for version '{self.version}'. "
                           f"Available splits: {list(split_mapping[self.version].keys())}")
        
        split_name = split_mapping[self.version][self.split]
        
        # Get blacklisted scenes (scenes without CAN bus data)
        blacklist = [419] + self.nusc_can.can_blacklist
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]
        
        # Get scenes for the split
        scenes = create_splits_scenes()[split_name][:]
        
        # Remove blacklisted scenes
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)
        
        return scenes

    def _filter_samples(self):
        """Filter samples to only include those from the specified scenes."""
        # Get all samples
        all_samples = [samp for samp in self.nusc.sample]
        
        # Filter samples that are in the current split's scenes
        filtered_samples = [
            samp for samp in all_samples 
            if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes
        ]
        
        # Sort by scene and timestamp for easier chronological processing
        filtered_samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        
        return filtered_samples

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def _resolve_sample_file_path(self, relative_path: str) -> str:
        """Resolve NuScenes sample_data filename across common dataroot layouts.

        Some installations keep camera files under "trainval/" or "test/"
        while metadata sits at the parent dataroot.
        """
        if os.path.isabs(relative_path):
            return relative_path

        candidates = [
            os.path.join(self.nusc.dataroot, relative_path),
            os.path.join(self.nusc.dataroot, "trainval", relative_path),
            os.path.join(self.nusc.dataroot, "test", relative_path),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return candidates[0]

    # only if we don't need image data, cache
    @conditional_decorator(cache, not GET_IMG_DATA)
    def __getitem__(self, idx):
        """
        Get item at the given index

        Returns:
            dict: A dictionary containing:
                - "sensor_data": dict of dicts with keys like "CAM_FRONT" containing
                  "img", "T_global_to_cam", "intrinsics"
                - "trajectory": future trajectory relative to ego vehicle as numpy array (N, 3)
                - "command": 1x3 one-hot encoded high-level command numpy array
                - "ego_trajectory": past ego trajectory as numpy array (past_frames, 3) with (x, y, θ)
                - "ego_velocity": current velocity as numpy array (3,) with (vx, vy, ω)
                - "ego_acceleration": current acceleration as numpy array (3,) with (ax, ay, β)
        """
        sample = self.samples[idx]

        if self.get_img_data:
            # Collect sensor data
            sensor_data = {}
            for sensor_name, sensor_token in sample["data"].items():
                # Only process camera sensors
                if "CAM" not in sensor_name:
                    continue

                sensor_dict = {}
                sensor_sample_data = self.nusc.get("sample_data", sensor_token)

                # Get image
                img_path = self._resolve_sample_file_path(sensor_sample_data["filename"])
                sensor_dict["img"] = np.array(Image.open(img_path))

                # Get calibration info
                calibrated_sensor = self.nusc.get(
                "calibrated_sensor", sensor_sample_data["calibrated_sensor_token"]
                )

                # Get intrinsic matrix
                sensor_dict["intrinsics"] = np.array(calibrated_sensor["camera_intrinsic"])

                # Get extrinsic transformation matrix (global to camera)
                rotation_quat = calibrated_sensor["rotation"]
                translation_vec = calibrated_sensor["translation"]

                # Convert rotation quaternion to rotation matrix
                rotation_matrix = Quaternion(rotation_quat).rotation_matrix

                # Build the transformation matrix (4x4)
                T_global_to_cam = np.eye(4)
                T_global_to_cam[:3, :3] = rotation_matrix
                T_global_to_cam[:3, 3] = translation_vec

                sensor_dict["T_global_to_cam"] = T_global_to_cam
                sensor_data[sensor_name] = sensor_dict
        else:
            sensor_data = {}

        # Get future trajectory in world frame
        trajectory_world = self._get_future_trajectory(sample)

        # Transform future way-points from world to ego vehicle frame
        current_lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose_0 = self.nusc.get("ego_pose", current_lidar_sd["ego_pose_token"])
        t0 = np.asarray(ego_pose_0["translation"])  # Current ego position in world
        q0 = Quaternion(ego_pose_0["rotation"])  # Current ego orientation in world (ego_frame_to_world)
        
        # Rotation matrix to transform from world frame to ego frame
        R_world_to_ego = q0.inverse.rotation_matrix

        if trajectory_world.size > 0:
            trajectory_local = (R_world_to_ego @ (trajectory_world - t0).T).T
        else: # Handle case of empty trajectory_world (e.g., if self.future_steps is 0)
            trajectory_local = np.empty((0, 3), dtype=np.float32)
            
        trajectory_local = trajectory_local.astype(np.float32)

        # Determine high-level command
        # Command is based on displacement at 3 seconds in the future.
        # Please note this is referenced from AD-MLP paper.
        # One-hot encoding: [Turn Left, Go Straight, Turn Right]
        one_hot_command = np.zeros((1, 3), dtype=np.float32)

        if self.future_steps > 0 and trajectory_local.shape[0] > 0:
            command_time_sec = 3.0
            # Calculate the target index for the 3-second mark.
            # Point i (0-indexed) is at time (i+1) / future_hz.
            # So, for time T, index = T * future_hz - 1.
            target_idx_for_command = int(command_time_sec * self.future_hz) - 1
            
            # Ensure target_idx_for_command is non-negative
            if target_idx_for_command < 0:
                target_idx_for_command = 0

            # Use the point at target_idx_for_command, or the last available point if shorter.
            actual_idx = min(target_idx_for_command, trajectory_local.shape[0] - 1)

            # Lateral displacement is the y-coordinate in the ego frame.
            # Assumes +y is to the left in ego frame.
            lateral_displacement = trajectory_local[actual_idx, 1]

            if lateral_displacement > 2.0:  # Turn Left
                one_hot_command[0, 0] = 1.0
            elif lateral_displacement < -2.0:  # Turn Right
                one_hot_command[0, 2] = 1.0
            else:  # Go Straight
                one_hot_command[0, 1] = 1.0
        else:
            # Default to "Go Straight" if no future trajectory is available
            one_hot_command[0, 1] = 1.0

        # get whether we are at a new sub-scene
        new_scene = (idx == 0) or \
                    (sample['scene_token'] != self.samples[idx-1]['scene_token'])
            
        # Get ego states
        ego_trajectory = self._get_past_ego_trajectory(sample)
        ego_velocity = self._get_ego_velocity(sample)
        ego_acceleration = self._get_ego_acceleration(sample)
            
        output = {
            "sensor_data": sensor_data,
            "trajectory": trajectory_local,
            "command": one_hot_command,
            "ego_trajectory": ego_trajectory,
            "ego_velocity": ego_velocity,
            "ego_acceleration": ego_acceleration,
            "scene_start": new_scene,
        }

        return output

    # @conditional_decorator(cache, GET_IMG_DATA) # cache only if we need image data
    def _get_future_trajectory(self, sample):
        """
        Extract the future trajectory of the ego vehicle in the world frame.

        Args:
            sample: Current sample

        Returns:
            np.ndarray: Array of shape (num_future_points, 3) containing future positions (x,y,z)
                        in the world frame. Returns an empty array of shape (0,3) if no future
                        points can be determined (e.g. self.future_steps is 0).
        """
        if self.future_steps == 0:
            return np.empty((0, 3), dtype=np.float32)

        trajectory_points = []
        current_sample_token = sample["token"]

        # Iterate through future samples
        num_points_collected = 0
        while num_points_collected < self.future_steps:
            if not sample["next"]: # No more samples
                break
            
            sample = self.nusc.get("sample", sample["next"])
            sample_data_lidar = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_pose = self.nusc.get("ego_pose", sample_data_lidar["ego_pose_token"])
            
            # Check if timestamps align with desired frequency (approximate)
            # This loop structure tries to get `self.future_steps` points.
            # For simplicity with the problem statement's fixed hz, we assume each 'next' sample
            # can be used, and rely on `self.future_steps` to cap.
            # A more robust implementation might check timestamps more carefully against `self.future_hz`.
            
            trajectory_points.append(ego_pose["translation"])
            num_points_collected += 1

        # If not enough future points were found, pad with the last known position
        if trajectory_points:
            last_pos = np.array(trajectory_points[-1])
            while len(trajectory_points) < self.future_steps:
                trajectory_points.append(last_pos.copy())
        else:
            # If no future points at all (e.g., end of a scene and self.future_steps > 0)
            # Pad with the current position.
            current_lidar_sd = self.nusc.get("sample_data", self.samples[self.__len__()-1 if sample["token"] == self.samples[self.__len__()-1]["token"] else 0]["data"]["LIDAR_TOP"]) # A bit hacky way to get a current sample if original 'sample' was already the last one
            # This fallback for 'current_pos' if trajectory_points is empty needs to be robust.
            # Using the initial sample's ego pose for this case.
            initial_sample_lidar_sd = self.nusc.get("sample_data", self.nusc.get("sample", current_sample_token)["data"]["LIDAR_TOP"])
            current_ego_pose = self.nusc.get("ego_pose", initial_sample_lidar_sd["ego_pose_token"])
            current_pos = np.array(current_ego_pose["translation"])
            for _ in range(self.future_steps):
                trajectory_points.append(current_pos.copy())
                
        return np.array(trajectory_points, dtype=np.float32)

    # @conditional_decorator(cache, GET_IMG_DATA) # cache only if we need image data
    def _get_past_ego_trajectory(self, sample):
        """
        Extract past ego trajectory (x, y, θ) for the past self.past_frames frames.
        
        Returns:
            np.ndarray: Array of shape (past_frames, 3) containing (x, y, θ)
        """
        trajectory_points = []
        current_sample = sample
        
        # Collect past samples (including current)
        samples_to_process = []
        for _ in range(self.past_frames):
            samples_to_process.append(current_sample)
            if current_sample["prev"]:
                current_sample = self.nusc.get("sample", current_sample["prev"])
            else:
                break
        
        # Reverse to get chronological order (oldest to newest)
        samples_to_process.reverse()
        
        # Extract trajectory for each sample
        for sample_item in samples_to_process:
            lidar_data = self.nusc.get("sample_data", sample_item["data"]["LIDAR_TOP"])
            ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
            
            x, y, z = ego_pose["translation"]
            # Convert quaternion to heading angle (θ)
            quat = Quaternion(ego_pose["rotation"])
            theta = quat.yaw_pitch_roll[0]  # yaw angle
            
            trajectory_points.append([x, y, theta])
        
        # Pad with the earliest point if not enough past samples
        while len(trajectory_points) < self.past_frames:
            trajectory_points.insert(0, trajectory_points[0].copy())
            
        return np.array(trajectory_points, dtype=np.float32)

    # @conditional_decorator(cache, GET_IMG_DATA) # cache regardless of GET_IMG_DATA
    def _get_ego_velocity(self, sample):
        """
        Extract ego velocity (vx, vy, ω) from CAN bus pose messages.
        
        Returns:
            np.ndarray: Array of shape (3,) containing [vx, vy, ω]
        """
        try:
            scene_name = self.nusc.get('scene', sample['scene_token'])['name']
            
            # Get pose messages from CAN bus (50Hz)
            pose_msgs = self.nusc_can.get_messages(scene_name, 'pose')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            
            # Get current sample timestamp
            ref_utime = sample['timestamp']
            
            if pose_msgs:
                # Find closest pose message
                pose_index = locate_message(pose_uts, ref_utime)
                pose_data = pose_msgs[pose_index]
                
                # Extract velocity (already in ego vehicle frame)
                vx, vy, vz = pose_data['vel']  # m/s
                
                # Extract angular velocity (rotation_rate in ego frame)
                omega_x, omega_y, omega_z = pose_data['rotation_rate']
                omega = omega_z  # Yaw rate (rad/s)
                
                return np.array([vx, vy, omega], dtype=np.float32)
            else:
                return np.zeros(3, dtype=np.float32)
                
        except Exception:
            return np.zeros(3, dtype=np.float32)

    # @conditional_decorator(cache, GET_IMG_DATA) # cache regardless of GET_IMG_DATA
    def _get_ego_acceleration(self, sample):
        """
        Extract ego acceleration (ax, ay, β) from CAN bus pose messages.
        
        Returns:
            np.ndarray: Array of shape (3,) containing [ax, ay, β]
        """
        try:
            scene_name = self.nusc.get('scene', sample['scene_token'])['name']
            
            # Get pose messages from CAN bus
            pose_msgs = self.nusc_can.get_messages(scene_name, 'pose')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            
            # Get current sample timestamp
            ref_utime = sample['timestamp']
            
            if pose_msgs:
                # Find closest pose message
                pose_index = locate_message(pose_uts, ref_utime)
                pose_data = pose_msgs[pose_index]
                
                # Extract linear acceleration (already in ego vehicle frame)
                ax, ay, az = pose_data['accel']  # m/s²
                
                # Calculate angular acceleration (β) from consecutive rotation rates
                beta = 0.0
                if pose_index > 0:
                    prev_pose = pose_msgs[pose_index - 1]
                    current_omega = pose_data['rotation_rate'][2]  # z-component (yaw rate)
                    prev_omega = prev_pose['rotation_rate'][2]
                    dt = (pose_data['utime'] - prev_pose['utime']) / 1e6  # Convert to seconds
                    
                    if dt > 0:
                        beta = (current_omega - prev_omega) / dt  # rad/s²
                
                return np.array([ax, ay, beta], dtype=np.float32)
            else:
                return np.zeros(3, dtype=np.float32)
                
        except Exception:
            return np.zeros(3, dtype=np.float32)