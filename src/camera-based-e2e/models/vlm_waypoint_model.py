"""
VLM-based waypoint prediction model.
Uses a frozen SmolVLM with a trainable MLP head for waypoint prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import os

class VLMWaypointModel(nn.Module):
    """
    Waypoint prediction model using frozen SmolVLM vision encoder.
    
    Architecture:
        1. Frozen SmolVLM vision encoder (2.1B params, bfloat16)
        2. Extract and pool visual features (1152-dim)
        3. Concatenate with past states and intent
        4. Trainable MLP head predicts waypoints
    """
    
    def __init__(
        self,
        model_name="HuggingFaceTB/SmolVLM-Base",
        in_dim=96,      # Past states: 16 * 6
        out_dim=40,     # Waypoints: 20 * 2
        hidden_dim=512, # MLP hidden size
        image_size=384  # Input image size for VLM
    ):
        super().__init__()
        
        print(f"Loading VLM: {model_name}")
        
        # Set environment variable for protobuf compatibility
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        # Load frozen VLM
        self.vlm = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Freeze all VLM parameters
        for param in self.vlm.parameters():
            param.requires_grad = False
        
        self.vlm.eval()  # Always in eval mode
        
        # Get vision hidden dimension
        self.vlm_dim = self.vlm.config.vision_config.hidden_size
        print(f"VLM vision hidden size: {self.vlm_dim}")
        
        self.image_size = image_size
        
        # Trainable waypoint prediction head
        # Input: [vision features (1152) + past (96) + intent (3)]
        input_dim = self.vlm_dim + in_dim + 3
        
        self.waypoint_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Print trainable params
        trainable = sum(p.numel() for p in self.waypoint_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.vlm.parameters())
        
        print(f"Model parameters:")
        print(f"  - Frozen VLM: {frozen:,}")
        print(f"  - Trainable head: {trainable:,}")
        print(f"  - Total: {total:,}")
        print(f"  - Trainable %: {100*trainable/total:.2f}%")
    
    def preprocess_images(self, images):
        """
        Preprocess images for VLM vision encoder.
        
        Args:
            images: (B, 3, H, W) tensor, typically (B, 3, 1280, 1920)
        
        Returns:
            (B, 3, image_size, image_size) tensor in bfloat16, normalized
        """
        # Convert to float if uint8
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        # Resize to VLM's expected input size
        if images.shape[2] != self.image_size or images.shape[3] != self.image_size:
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize with ImageNet stats (standard for vision models)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        # Convert to bfloat16 to match VLM weights
        images = images.to(torch.bfloat16)
        
        return images
    
    def extract_visual_features(self, images):
        """
        Extract visual features from VLM vision encoder.
        
        Args:
            images: (B, 3, image_size, image_size) in bfloat16
        
        Returns:
            (B, vlm_dim) pooled features in float32
        """
        with torch.no_grad():  # No gradients through frozen VLM
            # Forward through vision encoder
            outputs = self.vlm.vision_model(pixel_values=images)
            
            # Pool over spatial dimensions
            # outputs.last_hidden_state: (B, num_patches, hidden_size)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                visual_feat = outputs.pooler_output
            else:
                # Manual mean pooling over patches
                visual_feat = outputs.last_hidden_state.mean(dim=1)
        
        # Convert back to float32 for MLP head
        return visual_feat.float()
    
    def forward(self, x: dict):
        """
        Forward pass for waypoint prediction.
        
        Args:
            x: dict with keys:
                - 'PAST': (B, 16, 6) - past trajectory states
                - 'IMAGES': list of 6 images, each (B, 3, H, W)
                - 'INTENT': (B,) - intent signal (1, 2, or 3)
        
        Returns:
            (B, 40) - flattened waypoints to be reshaped as (B, 20, 2)
        """
        past = x['PAST']      # (B, 16, 6)
        images = x['IMAGES']  # list of 6 images
        intent = x['INTENT']  # (B,)
        
        # Get front camera (index 1 based on your loader)
        front_cam = images[1]  # (B, 3, H, W)
        
        # Preprocess images for VLM
        front_cam_processed = self.preprocess_images(front_cam)  # (B, 3, 384, 384), bfloat16
        
        # Extract visual features (frozen VLM)
        visual_feat = self.extract_visual_features(front_cam_processed)  # (B, 1152), float32
        
        # Prepare driving context
        # Intent: convert to one-hot (1,2,3 -> 0,1,2 indices)
        intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()  # (B, 3)
        
        # Past states: flatten
        past_flat = past.reshape(past.size(0), -1)  # (B, 96)
        
        # Concatenate all features
        combined = torch.cat([
            visual_feat,     # (B, 1152)
            past_flat,       # (B, 96)
            intent_onehot    # (B, 3)
        ], dim=1)  # (B, 1251)
        
        # Predict waypoints through trainable head
        waypoints = self.waypoint_head(combined)  # (B, 40)
        
        return waypoints


if __name__ == "__main__":
    # Quick test
    print("Testing VLMWaypointModel...")
    
    model = VLMWaypointModel(
        in_dim=96,
        out_dim=40,
        hidden_dim=512
    )
    
    # Create dummy inputs
    batch = {
        'PAST': torch.randn(2, 16, 6),
        'IMAGES': [torch.randint(0, 255, (2, 3, 1280, 1920), dtype=torch.uint8) for _ in range(6)],
        'INTENT': torch.tensor([1, 2])
    }
    
    # Forward pass
    print("\nTesting forward pass...")
    output = model(batch)
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 40)")
    
    if output.shape == (2, 40):
        print("\n✓ Model test passed!")
    else:
        print("\n✗ Model test failed!")
