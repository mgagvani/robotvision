"""
VLM vision encoder: split image into patches, encode each independently, concat.

Backbone: the per-patch encoder. It receives a single patch (B, 3, patch_size, patch_size)
and returns (B, N, C) or (B, C). Typical choices:
  - timm ViT (e.g. vit_tiny_patch16_224, vit_small_patch16_224) via build_timm_patch_encoder
  - CLIP vision encoder (e.g. open_clip or LLaVA-NeXT CLIP ViT-L-336)
  - Any nn.Module with .embed_dim or .num_features that accepts (B, 3, H, W)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import timm

class VisionEncoder(nn.Module):
    """
    Vision encoder: split image into patches (patch_size x patch_size),
    encode each with a backbone, concatenate along sequence dim -> (B, N, C).

    Backbone: per-patch encoder (e.g. timm ViT, CLIP, LLaVA-NeXT vision encoder).
    Must accept (B, 3, H, W) and return (B, N, C) or (B, C), and have .embed_dim or .num_features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_size: int,
        num_patches: int = 2,
        frozen: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features", None))
        if self.embed_dim is None:
            raise ValueError("backbone must have .embed_dim or .num_features")
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W), uint8 or float. Split into num_patches of (patch_size, patch_size), encode each, concat.
        Returns: (B, N, embed_dim).
        """
        if x.dtype != torch.float32:
            x = x.to(torch.float32) / 255.0
        patches = self._split(x)
        feats: List[torch.Tensor] = []
        for patch in patches:
            out = self.backbone(patch)
            if out.dim() == 2:
                out = out.unsqueeze(1)
            feats.append(out)
        return torch.cat(feats, dim=1)

    def _split(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Split image into num_patches crops of (patch_size, patch_size) along the longer side.
        If image is smaller than patch_size, return a single patch (resized). Each crop is
        resized to (patch_size, patch_size) so the backbone sees a fixed input size.
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        patches: List[torch.Tensor] = []
        if H <= ps and W <= ps:
            p = _resize_to(x, ps, ps)
            return [p]
        n = self.num_patches
        if W >= H and W > ps:
            step = max(1, (W - ps) // max(1, n - 1))
            for i in range(n):
                start = min(i * step, W - ps)
                crop = x[:, :, :, start : start + ps]
                patches.append(_resize_to(crop, ps, ps))
        else:
            step = max(1, (H - ps) // max(1, n - 1))
            for i in range(n):
                start = min(i * step, H - ps)
                crop = x[:, :, start : start + ps, :]
                patches.append(_resize_to(crop, ps, ps))
        return patches


def _resize_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Resize tensor last two dims to (h, w) with bilinear interpolation.
    No-op if already (h, w). Used to normalize patch size for the backbone.
    """
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def build_timm_patch_encoder(
    model_name: str = "vit_tiny_patch16_224",
    pretrained: bool = True,
    global_pool: str = "",
    num_classes: int = 0,
) -> nn.Module:
    """
    Build a timm model suitable as VisionEncoder backbone (per-patch encoder).
    Returns a model that takes (B, 3, H, W) and outputs (B, N, C) with .embed_dim set.
    Example backbones: vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        global_pool=global_pool,
    )
    
    model.embed_dim = (
        getattr(model, "embed_dim", None)
        or getattr(model, "num_features", None)
        or (getattr(model.head, "in_features", None) if hasattr(model, "head") and model.head is not None and not isinstance(model.head, nn.Identity) else None)
    )
    if model.embed_dim is None:
        model.embed_dim = 192
    return model
