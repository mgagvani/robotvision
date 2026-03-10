import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
import cv2

# Add the parent directory to path to import TwinLiteNetPlus
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src" / "camera-based-e2e" / "losses"))

from losses.TwinLiteNetPlus.model import TwinLiteNetPlus
from losses.TwinLiteNetPlus import config as twinlite_config

# Model configuration - using medium as default (pretrained weights available)
DEFAULT_CONFIG = "medium"
MODEL_WEIGHTS_PATH = str(current_dir / "TwinLiteNetPlus" / "medium.pth")


def encode_lanes(lanes):
    """
    Encode lane detection results for storage.
    
    Args:
        lanes: Lane detection tensor of shape (B, 2, H, W) or dict with 'da_seg'/'ll_seg'
              
    Returns:
        Encoded lane detection data (picklable)
    """
    if isinstance(lanes, dict):
        # If it's a dict from get_lane_lines, convert to storable format
        encoded = {}
        for key, value in lanes.items():
            if isinstance(value, torch.Tensor):
                encoded[key] = value.cpu()
            else:
                encoded[key] = value
        return encoded
    elif isinstance(lanes, torch.Tensor):
        return lanes.cpu()
    else:
        return lanes


class LaneDetectionLoss:
    def __init__(self, device, config=DEFAULT_CONFIG, use_fp16=True):
        """
        Initialize the lane detection loss function using TwinLiteNetPlus.
        
        Args:
            device: torch device (cuda or cpu)
            config: model configuration ("nano", "small", "medium", "large")
            use_fp16: Use FP16 (half precision) for faster inference on GPU
        """
        self.device = device
        self.config = config
        # Handle both string and torch.device types
        if isinstance(device, str):
            self.use_fp16 = use_fp16 and device == 'cuda'
        else:
            self.use_fp16 = use_fp16 and device.type == 'cuda'
        
        # Create args-like object for the model
        class Args:
            def __init__(self, config):
                self.config = config
        
        args = Args(config)
        
        # Initialize the TwinLiteNetPlus model
        self.lane_model = TwinLiteNetPlus(args).to(self.device)
        
        # Load pretrained weights
        try:
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=self.device)
            self.lane_model.load_state_dict(state_dict)
            print(f"Loaded TwinLiteNetPlus weights from {MODEL_WEIGHTS_PATH}")
        except FileNotFoundError:
            print(f"Warning: Could not find weights at {MODEL_WEIGHTS_PATH}, using random initialization")
        
        # Use FP16 for faster inference
        if self.use_fp16:
            self.lane_model.half()
        
        self.lane_model.eval()
        
        # Image preprocessing parameters (from demo.py)
        self.img_size = 640
        
    def preprocess_images(self, images):
        """
        Preprocess images for TwinLiteNetPlus model.
        
        Args:
            images: torch.Tensor of shape (B, C, H, W) with values in [0, 255] or [0, 1]
            
        Returns:
            Preprocessed images ready for the model
        """
        # Ensure images are in [0, 1] range
        if images.max() > 1.0:
            images = images / 255.0
        
        # Get original dimensions
        batch_size, c, height, width = images.shape
        
        # Use padding that is compatible with the model's expected input
        # The model expects 640x640, so we resize
        resized = F.interpolate(images, size=(self.img_size, self.img_size), 
                                mode='bilinear', align_corners=False)
        
        # Convert to half precision if using FP16
        if self.use_fp16:
            resized = resized.half()
        
        return resized, (height, width)
    
    def get_lane_lines(self, images, output_size=None):
        """
        Compute ground truth lane lines from images using TwinLiteNetPlus.
        
        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W). Values in [0, 255] or [0, 1].
            output_size: Optional tuple (H, W) to resize output to. If None, uses original image size.
            
        Returns:
            dict with keys:
                - 'da_seg': Driving area segmentation logits (B, 2, H, W)
                - 'll_seg': Lane line segmentation logits (B, 2, H, W)
                - 'da_mask': Driving area binary mask (B, H, W) 
                - 'll_mask': Lane line binary mask (B, H, W)
        """
        # Preprocess images
        preprocessed, original_size = self.preprocess_images(images)
        
        # Forward pass through the lane detection model
        with torch.no_grad():
            da_seg_out, ll_seg_out = self.lane_model(preprocessed)
        
        # Get original dimensions
        orig_h, orig_w = original_size
        
        # Interpolate to output size
        target_size = output_size if output_size is not None else (orig_h, orig_w)
        da_seg_out = F.interpolate(da_seg_out, size=target_size, mode='bilinear', align_corners=False)
        ll_seg_out = F.interpolate(ll_seg_out, size=target_size, mode='bilinear', align_corners=False)
        
        # Convert back to float32 if using FP16 for argmax
        if self.use_fp16:
            da_seg_out = da_seg_out.float()
            ll_seg_out = ll_seg_out.float()
        
        # Get binary masks by taking argmax
        da_preds = torch.argmax(da_seg_out, dim=1)  # (B, H, W)
        ll_preds = torch.argmax(ll_seg_out, dim=1)  # (B, H, W)
        
        return {
            'da_seg': da_seg_out,  # (B, 2, H, W) - class logits
            'll_seg': ll_seg_out,  # (B, 2, H, W) - class logits
            'da_mask': da_preds,   # (B, H, W) - binary mask
            'll_mask': ll_preds,   # (B, H, W) - binary mask
        }
    
    def get_lanes(self, images, output_size=None):
        """
        Alias for get_lane_lines to match compute.py API.
        
        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W).
            output_size: Optional tuple (H, W) to resize output to.
            
        Returns:
            Lane detection tensor (B, 2, H, W) - lane line segmentation logits
        """
        result = self.get_lane_lines(images, output_size)
        # Return ll_seg (lane line) as the primary output for road line detection
        return result['ll_seg']
    
    def compute_lane_loss(self, gt_images, pred_lane_logits, loss_fn=F.binary_cross_entropy_with_logits):
        """
        Compute lane detection loss.
        
        Args:
            gt_images: Ground truth images to compute lane detection from
            pred_lane_logits: Predicted lane segmentation logits from the model
            loss_fn: Loss function to use
            
        Returns:
            Lane detection loss (scalar tensor)
        """
        # Get ground truth lane lines from images using TwinLiteNetPlus
        gt_lane = self.get_lane_lines(gt_images, output_size=(pred_lane_logits.shape[2], pred_lane_logits.shape[3]))
        
        # Use lane line segmentation (ll_seg) for road line detection
        gt_ll_seg = gt_lane['ll_seg']  # (B, 2, H, W)
        
        # Compute loss between predicted and ground truth lane line logits
        lane_loss = loss_fn(pred_lane_logits, gt_ll_seg)
        
        return lane_loss
    
    def compute_combined_loss(self, gt_images, pred_da_logits, pred_ll_logits, 
                               da_weight=0.5, ll_weight=0.5, loss_fn=F.binary_cross_entropy_with_logits):
        """
        Compute combined driving area and lane line loss.
        
        Args:
            gt_images: Ground truth images
            pred_da_logits: Predicted driving area segmentation logits
            pred_ll_logits: Predicted lane line segmentation logits
            da_weight: Weight for driving area loss
            ll_weight: Weight for lane line loss
            loss_fn: Loss function to use
            
        Returns:
            Combined loss (scalar tensor)
        """
        # Get ground truth from images
        gt_lane = self.get_lane_lines(gt_images, output_size=(pred_da_logits.shape[2], pred_da_logits.shape[3]))
        
        gt_da_seg = gt_lane['da_seg']
        gt_ll_seg = gt_lane['ll_seg']
        
        # Compute individual losses
        da_loss = loss_fn(pred_da_logits, gt_da_seg)
        ll_loss = loss_fn(pred_ll_logits, gt_ll_seg)
        
        # Combined loss
        combined_loss = da_weight * da_loss + ll_weight * ll_loss
        
        return combined_loss
    
    def __call__(self, gt_images, pred_lanes, loss_fn=F.binary_cross_entropy_with_logits):
        """
        Main call method for computing lane detection loss.
        
        Args:
            gt_images: Ground truth images
            pred_lanes: Predicted lane segmentation (can be dict with 'da_seg' and 'll_seg', or just logits)
            loss_fn: Loss function to use
            
        Returns:
            Lane detection loss
        """
        if isinstance(pred_lanes, dict):
            # If pred_lanes is a dict, compute combined loss
            pred_da = pred_lanes.get('da_seg')
            pred_ll = pred_lanes.get('ll_seg')
            
            if pred_da is not None and pred_ll is not None:
                return self.compute_combined_loss(gt_images, pred_da, pred_ll, loss_fn=loss_fn)
            elif pred_ll is not None:
                return self.compute_lane_loss(gt_images, pred_ll, loss_fn=loss_fn)
            else:
                raise ValueError("pred_lanes dict must contain 'da_seg' or 'll_seg'")
        else:
            # Assume pred_lanes is just the lane line logits
            return self.compute_lane_loss(gt_images, pred_lanes, loss_fn=loss_fn)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Must run on GPU")
        exit(1)

    import sys
    from pathlib import Path
    from matplotlib import pyplot as plt
    
    # Add project root to path for importing loader
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    sys.path.append(str(project_root))
    
    from loader import WaymoE2E
    
    # Load Waymo dataset
    loader = WaymoE2E(indexFile="index_val.pkl", data_dir="/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/")
    data_iterator = iter(torch.utils.data.DataLoader(loader, batch_size=8, num_workers=4))
    
    device = torch.device("cuda")
    lane_loss_fn = LaneDetectionLoss(device)
    
    # Skip some batches to get interesting data
    for _ in range(6):
        batch = next(data_iterator)
    
    images = batch["IMAGES"][1].to(device)  # front camera
    
    # Get lane detections
    lanes = lane_loss_fn.get_lane_lines(images)
    
    print(f"DA Seg shape: {lanes['da_seg'].shape}")
    print(f"LL Seg shape: {lanes['ll_seg'].shape}")
    print(f"DA Mask shape: {lanes['da_mask'].shape}")
    print(f"LL Mask shape: {lanes['ll_mask'].shape}")
    
    # Visualize results
    fig, ax = plt.subplots(8, 3, figsize=(16, 48))
    for i in range(8):
        # Input image
        ax[i, 0].set_title(f"Input Image {i+1}")
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # Handle both [0, 1] and [0, 255] ranges
        if img.max() > 1.0:
            img = img.astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8)
        ax[i, 0].imshow(img)
        ax[i, 0].axis('off')
        
        # Driving Area Segmentation
        ax[i, 1].set_title(f"Driving Area {i+1}")
        da_mask = lanes['da_mask'][i].cpu().numpy()
        ax[i, 1].imshow(da_mask, cmap='gray')
        ax[i, 1].axis('off')
        
        # Lane Line Segmentation
        ax[i, 2].set_title(f"Lane Lines {i+1}")
        ll_mask = lanes['ll_mask'][i].cpu().numpy()
        ax[i, 2].imshow(ll_mask, cmap='gray')
        ax[i, 2].axis('off')
    
    plt.tight_layout()
    fig.savefig("lane_detections.png", dpi=150, bbox_inches='tight')
    print("Saved lane_detections.png")
    
    # Throughput test
    from time import perf_counter
    
    start_time = perf_counter()
    times = []
    for _ in range(100):
        batch = next(data_iterator)
        images = batch["IMAGES"][1].to(device)  # front camera
        t0 = perf_counter()
        lanes = lane_loss_fn.get_lane_lines(images)
        times.append(perf_counter() - t0)
    end_time = perf_counter()
    
    print(f"Throughput: {100 / (end_time - start_time):.2f} batches/sec")
    print(f"Avg lane inference: {1 / np.mean(times):.2f} FPS")
