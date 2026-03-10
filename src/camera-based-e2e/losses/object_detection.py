import torch
import torch.nn.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np
from typing import Dict, List, Optional, Tuple
from rfdetr import RFDETRBase


OBJ_DET_MODEL_ID = "facebook/detr-resnet-50"

# Common detection classes for autonomous driving
DETECTION_CLASSES = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

class ObjectDetectionLoss:
    """
    Object Detection Loss for computing ground truth detections from images.
    Uses a DETR-based model for real-time object detection.
    """
    
    def __init__(self, device, confidence_threshold: float = 0.7):
        """
        Initialize the object detection model.
        
        Args:
            device: torch device (cuda/cpu)
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.processor = DetrImageProcessor.from_pretrained(OBJ_DET_MODEL_ID)
        self.model = DetrForObjectDetection.from_pretrained(OBJ_DET_MODEL_ID).to(self.device)
        self.model.eval()
        
        # COCO class IDs - DETR was trained on COCO
        # Mapping to driving-relevant classes
        self.class_mapping = {
            0: 'person',      # person
            1: 'bicycle',     # bicycle  
            2: 'car',        # car
            3: 'motorcycle', # motorcycle
            5: 'bus',        # bus
            7: 'truck',      # truck
            9: 'traffic light',
            11: 'stop sign',
            
        }
    
    def get_detections(self, images: torch.Tensor) -> List[Dict]:
        """
        Compute object detections from images.
        
        Args:
            images: Input images of shape (B, C, H, W)
            
        Returns:
            List of detection dictionaries, one per image:
            {
                'boxes': Tensor of shape (N, 4) in [x1, y1, x2, y2] format
                'labels': Tensor of shape (N,) with class labels
                'scores': Tensor of shape (N,) with confidence scores
            }
        """
        # Prepare inputs
        images_np = images.cpu().numpy()
        batch_size = images.shape[0]
        
        all_detections = []
        
        for i in range(batch_size):
            # Convert to PIL-like format for processor
            img = images_np[i].transpose(1, 2, 0)  # (H, W, C)
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            
            # Process image
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get results
            target_sizes = torch.tensor([img.shape[:2]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]

            # Filter and map classes
            scores = results['scores']
            labels = results['labels']
            boxes = results['boxes']
            
            # Filter by confidence
            mask = scores > self.confidence_threshold
            scores = scores[mask]
            labels = labels[mask]
            boxes = boxes[mask]
            
            # Map to driving classes (0-7)
            mapped_labels = []
            for label in labels:
                if label.item() in self.class_mapping:
                    # Map to our classes: person=0, rider=1, car=2, truck=3, bus=4, motorcycle=5, bicycle=6
                    coco_id = label.item()
                    if coco_id == 0:
                        mapped_labels.append(0)  # person
                    elif coco_id == 2:
                        mapped_labels.append(2)  # car
                    elif coco_id == 3:
                        mapped_labels.append(6)  # motorcycle
                    elif coco_id == 5:
                        mapped_labels.append(4)  # bus
                    elif coco_id == 7:
                        mapped_labels.append(3)  # truck
                    else:
                        mapped_labels.append(-1)  # unknown
                else:
                    mapped_labels.append(-1)
            
            detection = {
                'boxes': boxes.cpu(),
                'labels': torch.tensor(mapped_labels, device='cpu') if mapped_labels else torch.tensor([], dtype=torch.long),
                'scores': scores.cpu()
            }
            all_detections.append(detection)
        
        return all_detections
    
    def compute_detection_loss(
        self, 
        gt_images: torch.Tensor, 
        pred_detections: List[Dict], 
        loss_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Compute object detection loss between predicted and ground truth detections.
        
        Args:
            gt_images: Ground truth images (B, C, H, W)
            pred_detections: Predicted detections from the model
            loss_fn: Optional custom loss function
            
        Returns:
            Detection loss tensor
        """
        # Get ground truth detections
        gt_detections = self.get_detections(gt_images)
        
        if loss_fn is None:
            # Default: compute confidence-based loss
            loss = torch.tensor(0.0, device=self.device)
            
            for gt, pred in zip(gt_detections, pred_detections):
                if len(gt['scores']) > 0 and len(pred['scores']) > 0:
                    # Score loss
                    score_loss = F.mse_loss(pred['scores'], gt['scores'])
                    loss = loss + score_loss
                    
                    # Box loss (if both have detections)
                    if len(pred['boxes']) > 0:
                        box_loss = F.l1_loss(pred['boxes'], gt['boxes'])
                        loss = loss + box_loss
                elif len(gt['scores']) > 0:
                    # Penalize missing detections
                    loss = loss + len(gt['scores']) * 0.1
                elif len(pred['scores']) > 0:
                    # Penalize false positives
                    loss = loss + len(pred['scores']) * 0.1
            
            loss = loss / max(len(gt_detections), 1)
        else:
            loss = loss_fn(gt_detections, pred_detections)
        
        return loss
    
    def __call__(
        self, 
        gt_images: torch.Tensor, 
        pred_detections: List[Dict], 
        loss_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """Convenience method to compute detection loss."""
        return self.compute_detection_loss(gt_images, pred_detections, loss_fn)


def encode_detections(detections: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Encode detections to a format suitable for storage.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Encoded dictionary with tensors
    """
    batch_size = len(detections)
    max_detections = max(len(d['boxes']) for d in detections) if detections else 0
    
    # Initialize tensors
    boxes = torch.zeros(batch_size, max_detections, 4)
    labels = torch.full((batch_size, max_detections), -1, dtype=torch.long)
    scores = torch.zeros(batch_size, max_detections)
    
    for i, det in enumerate(detections):
        n = len(det['boxes'])
        if n > 0:
            boxes[i, :n] = det['boxes']
            labels[i, :n] = det['labels']
            scores[i, :n] = det['scores']
    
    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }


def decode_detections(encoded: Dict[str, torch.Tensor]) -> List[Dict]:
    """
    Decode stored detections back to list format.
    
    Args:
        encoded: Encoded detection dictionary
        
    Returns:
        List of detection dictionaries
    """
    batch_size = encoded['boxes'].shape[0]
    detections = []
    
    for i in range(batch_size):
        mask = encoded['scores'][i] > 0
        det = {
            'boxes': encoded['boxes'][i, mask],
            'labels': encoded['labels'][i, mask],
            'scores': encoded['scores'][i, mask]
        }
        detections.append(det)
    
    return detections


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Must run on GPU")
    
    import sys
    from pathlib import Path
    from matplotlib import pyplot as plt
    
    # Add project root to path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    sys.path.append(str(project_root))
    
    from loader import WaymoE2E
    
    # Load data
    loader = WaymoE2E(
        indexFile="index_val.pkl",
        data_dir="/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/"
    )
    data_iterator = iter(torch.utils.data.DataLoader(loader, batch_size=4, num_workers=2))
    
    device = torch.device("cuda")
    obj_det_loss = ObjectDetectionLoss(device)
    
    for _ in range(3):
        batch = next(data_iterator)
    
    images = batch["IMAGES"][1].to(device)  # front camera
    
    print("Computing object detections...")
    detections = obj_det_loss.get_detections(images)
    
    # Print detection summary
    for i, det in enumerate(detections):
        print(f"Image {i}: {len(det['boxes'])} detections")
        if len(det['labels']) > 0:
            unique_labels = torch.unique(det['labels'])
            print(f"  Classes: {unique_labels.tolist()}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i in range(4):
        ax = axes[i // 2, i % 2]
        img = images[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"Image {i+1}")
        
        # Draw boxes
        det = detections[i]
        for box, label, score in zip(det['boxes'], det['labels'], det['scores']):
            x1, y1, x2, y2 = box.numpy()
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color='red', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x1, y1, f"{label.item()}: {score:.2f}", 
                   color='red', fontsize=8, backgroundcolor='white')
        ax.axis('off')
    
    plt.tight_layout()
    fig.savefig("object_detections.png", dpi=150)
    print("Saved object_detections.png")
