"""
Multi-camera scene encoder for proposal-centric E2E planner.
Encodes all 8 Waymo cameras with a shared backbone and fuses via concatenation + camera embeddings.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneEncoder(nn.Module):
    """
    Encode multi-camera images into a single scene feature sequence.
    Uses a shared backbone per camera, then concatenates tokens and adds camera embeddings.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 256,
        num_cams: int = 8,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_cams = num_cams
        # Backbone output dim (from feature extractor .dims or .feature_dim)
        if hasattr(backbone, "dims"):
            self.backbone_dim = sum(backbone.dims)
        else:
            self.backbone_dim = getattr(backbone, "feature_dim", 384)

        self.proj = nn.Linear(self.backbone_dim, d_model)
        # Per-camera embedding (1, num_cams, d_model) added to all tokens of that camera
        self.cam_embed = nn.Parameter(torch.zeros(1, num_cams, d_model))
        nn.init.trunc_normal_(self.cam_embed, std=0.02)
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: List of (B, C, H, W) tensors, one per camera. len(images) == num_cams.
        Returns:
            scene_feat: (B, N, d_model) with N = num_cams * n_tokens_per_cam
        """
        B = images[0].size(0)
        all_tokens = []
        for c, img in enumerate(images):
            with torch.no_grad():
                feats = self.backbone(img)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]  # (B, C, H, W)
            # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            tokens = feats.flatten(2).permute(0, 2, 1)  # (B, n_tokens, backbone_dim)
            tokens = self.proj(tokens) + self.cam_embed[:, c : c + 1, :]  # (B, n_tokens, d_model)
            all_tokens.append(tokens)
        # (B, num_cams * n_tokens_per_cam, d_model)
        scene = torch.cat(all_tokens, dim=1)
        return self.ln(scene)
