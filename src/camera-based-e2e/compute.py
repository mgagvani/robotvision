"""
Precomputation Script for Depth, Object Detection, and Lane Detection

This script computes and stores precomputed ground truth data:
- Depth maps using Depth-Anything-V2
- Object detections using DETR (RF-DETR style)
- Lane detections using TwinLiteNet

The precomputed data is saved in shards for efficient loading during training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle
import gc

# Add project root to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from loader import WaymoE2E
from losses.depth import DepthLoss
from losses.object_detection import ObjectDetectionLoss, encode_detections
from losses.lane_detection import LaneDetectionLoss, encode_lanes
from models.monocular import SAMFeatures


def precompute_perception(
    data_dir: str,
    index_file: str,
    output_dir: str,
    shard_size: int = 1000,
    batch_size: int = 8,
    num_workers: int = 4,
    cameras: list = None,
    device: str = "cuda"
):
    """
    Precompute depth, object detection, and lane detection for the dataset.
    
    Each shard contains shard_size samples.
    Each sample has keys: tokens, depth, obj_det, lane_det - each containing a list of 3 (for 3 cameras).
    
    Args:
        data_dir: Path to the Waymo dataset
        index_file: Path to the index file
        output_dir: Output directory for precomputed shards
        shard_size: Number of samples per shard
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        cameras: List of camera indices to process (default: [0,1,2] = front-left, front, front-right)
        device: Device to run inference on
    """
    if cameras is None:
        cameras = [0, 1, 2]  # front-left, front, front-right
    
    # Initialize the perception models
    print("Initializing depth model...")
    depth_model = DepthLoss(device)
    
    print("Initializing object detection model...")
    obj_det_model = ObjectDetectionLoss(device, confidence_threshold=0.5)
    
    print("Initializing lane detection model...")
    lane_det_model = LaneDetectionLoss(device)
    
    print("Initializing SAM features extractor...")
    visual_feature_extractor = SAMFeatures(model_name="timm/vit_pe_spatial_tiny_patch16_512.fb")
    visual_feature_extractor = visual_feature_extractor.to(device)
    visual_feature_extractor.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset - single dataset and dataloader for all shards
    print(f"Loading dataset from {index_file}...")
    dataset = WaymoE2E(
        indexFile=index_file,
        data_dir=data_dir,
        use_precomputed=False,
        shard_size=shard_size
    )
    
    # Use num_workers=0 for deterministic ordering without worker splitting
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Process all data with a single dataloader
    print(f"Processing dataset and saving shards to {output_dir}...")
    
    shard_idx = 0
    current_shard = []  # List of sample dicts
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        images_list = batch["IMAGES"]  # List of lists [camera][batch]
        batch_size_curr = images_list[0].shape[0]  # Get actual batch size
        
        # Process each sample in the batch
        for sample_idx in range(batch_size_curr):
            # Create a sample dict with data for all cameras
            sample_data = {
                'tokens': [],
                'depth': [],
                'obj_det': [],
                'lane_det': []
            }
            
            # Process each camera
            for cam_idx in cameras:
                if cam_idx >= len(images_list):
                    continue
                
                # Get this sample's image for this camera
                image = images_list[cam_idx][sample_idx:sample_idx+1].to(device)
                
                # Compute depth
                with torch.no_grad():
                    depth = depth_model.get_depth(image)
                    depth = F.interpolate(depth.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                
                # Compute object detections
                with torch.no_grad():
                    detections = obj_det_model.get_detections(image)
                    detections = encode_detections(detections)
                
                # Compute lane detection
                with torch.no_grad():
                    lanes = lane_det_model.get_lanes(image)
                    lanes = F.interpolate(lanes, size=(128, 128), mode='bilinear', align_corners=False)
                
                # Compute visual tokens
                with torch.no_grad():
                    feats = visual_feature_extractor(image)
                    if isinstance(feats, (list, tuple)):
                        tokens = torch.cat([f.flatten(2) for f in feats], dim=1)
                    else:
                        tokens = feats.flatten(2)
                    tokens = torch.permute(tokens, (0, 2, 1)).squeeze(0)
                
                # Append camera data to sample
                sample_data['tokens'].append(tokens.cpu())
                sample_data['depth'].append(depth.cpu())
                sample_data['obj_det'].append(detections)
                sample_data['lane_det'].append(lanes.cpu())
            
            current_shard.append(sample_data)
            
            # If we've collected shard_size samples, save the shard
            if len(current_shard) >= shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
                torch.save(current_shard, shard_path)
                print(f"Saved shard {shard_idx} with {len(current_shard)} samples")
                shard_idx += 1
                current_shard = []
    
    # Save any remaining samples as the last shard
    if len(current_shard) > 0:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(current_shard, shard_path)
        print(f"Saved final shard {shard_idx} with {len(current_shard)} samples")
    
    print(f"Precomputation complete. Saved {shard_idx + 1} shards to {output_dir}")

class PerceptionLoss:
    """
    Combined perception loss that computes depth, object detection, and lane detection losses.
    This class can be used during training to compute losses against precomputed ground truth.
    """
    
    def __init__(self, device, depth_weight: float = 1.0, obj_det_weight: float = 1.0, lane_weight: float = 1.0):
        """
        Initialize the combined perception loss.
        
        Args:
            device: torch device
            depth_weight: Weight for depth loss
            obj_det_weight: Weight for object detection loss
            lane_weight: Weight for lane detection loss
        """
        self.device = device
        self.depth_weight = depth_weight
        self.obj_det_weight = obj_det_weight
        self.lane_weight = lane_weight
        
        # Initialize models for computing ground truth
        print("Initializing depth model for training...")
        self.depth_model = DepthLoss(device)
        
        print("Initializing object detection model for training...")
        self.obj_det_model = ObjectDetectionLoss(device)
        
        print("Initializing lane detection model for training...")
        self.lane_det_model = LaneDetectionLoss(device)
    
    def compute_depth_loss(self, gt_images: torch.Tensor, pred_depths: torch.Tensor) -> torch.Tensor:
        """Compute depth loss."""
        gt_depth = self.depth_model.get_depth(gt_images)
        loss = torch.nn.functional.l1_loss(pred_depths, gt_depth)
        return loss * self.depth_weight
    
    def compute_obj_det_loss(self, gt_images: torch.Tensor, pred_detections: list) -> torch.Tensor:
        """Compute object detection loss."""
        gt_detections = self.obj_det_model.get_detections(gt_images)
        loss = self.obj_det_model.compute_detection_loss(gt_images, pred_detections)
        return loss * self.obj_det_weight
    
    def compute_lane_loss(self, gt_images: torch.Tensor, pred_lanes: torch.Tensor) -> torch.Tensor:
        """Compute lane detection loss."""
        gt_lanes = self.lane_det_model.get_lanes(gt_images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_lanes[:, 0, :, :],
            gt_lanes[:, 0, :, :]
        )
        return loss * self.lane_weight
    
    def __call__(
        self,
        gt_images: torch.Tensor,
        pred_depths: torch.Tensor = None,
        pred_detections: list = None,
        pred_lanes: torch.Tensor = None
    ) -> dict:
        """
        Compute all perception losses.
        
        Args:
            gt_images: Ground truth images
            pred_depths: Predicted depth maps (optional)
            pred_detections: Predicted object detections (optional)
            pred_lanes: Predicted lane segmentation (optional)
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        if pred_depths is not None:
            losses['depth'] = self.compute_depth_loss(gt_images, pred_depths)
            total_loss = total_loss + losses['depth']
        
        if pred_detections is not None:
            losses['obj_det'] = self.compute_obj_det_loss(gt_images, pred_detections)
            total_loss = total_loss + losses['obj_det']
        
        if pred_lanes is not None:
            losses['lane'] = self.compute_lane_loss(gt_images, pred_lanes)
            total_loss = total_loss + losses['lane']
        
        losses['total'] = total_loss
        return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute perception data")
    parser.add_argument("--data_dir", type=str, default="/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/",
                        help="Path to Waymo dataset")
    parser.add_argument("--index_file", type=str, default="index_train.pkl",
                        help="Path to index file")
    parser.add_argument("--output_dir", type=str, default="/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/precomputed",
                        help="Output directory for precomputed shards")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Number of samples per shard")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()

    
    precompute_perception(
        args.data_dir,
        args.index_file,
        args.output_dir,
        args.shard_size,
        args.batch_size,
        args.num_workers,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
