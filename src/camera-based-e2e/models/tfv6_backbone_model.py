from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the LEAD package (models/lead/lead) is importable when this script is run
# from src/camera-based-e2e without an editable install.
_LEAD_PACKAGE_ROOT = Path(__file__).resolve().parent / "lead_ltf"
if str(_LEAD_PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(_LEAD_PACKAGE_ROOT))

from lead.tfv6.tfv6 import TFv6  # noqa: E402
from lead.training.config_training import TrainingConfig  # noqa: E402


class TFv6BackboneTrajectoryModel(nn.Module):
    """
    Trajectory head on top of LEAD TFv6 Transfuser backbone.

    This model matches the `BaseModel`/`LitModel` training contract:
    input is a dict with `PAST`, `IMAGES`, `INTENT`, output is `(B, T*2)`.
    """

    def __init__(self, out_dim: int, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.out_dim = out_dim

        config = TrainingConfig()
        config.use_waymo_e2e_data = True
        config.use_carla_data = False
        config.use_navsim_data = False
        config.use_semantic = False
        config.use_bev_semantic = False
        config.detect_boxes = False
        config.use_planning_decoder = False
        config.LTF = True
        config.freeze_backbone = freeze_backbone
        self.config = config

        # Build backbone via TFv6 so this model follows TFv6 construction.
        tfv6 = TFv6(device=torch.device("cpu"), config=self.config)
        self.backbone = tfv6.backbone

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.bev_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.past_encoder = nn.Sequential(
            nn.Linear(16 * 6, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.intent_embedding = nn.Embedding(num_embeddings=8, embedding_dim=16)
        self.head = nn.Sequential(
            nn.Linear(self.config.bev_features_chanels + 128 + 16, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def _select_three_cameras(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        # Waymo ordering in this project uses front-left, front, front-right first.
        if len(images) >= 3:
            return images[:3]
        if len(images) == 2:
            return [images[0], images[1], images[1]]
        if len(images) == 1:
            return [images[0], images[0], images[0]]
        raise ValueError("Expected at least one camera tensor in `IMAGES`.")

    def _build_rgb_tensor(self, images: list[torch.Tensor]) -> torch.Tensor:
        selected = self._select_three_cameras(images)

        target_h = self.config.final_image_height
        target_w = self.config.camera_calibration[1]["width"]
        resized = []
        for cam in selected:
            cam = cam.float()
            if cam.max() > 1.0:
                cam = cam / 255.0
            resized.append(
                F.interpolate(
                    cam,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(resized, dim=-1)

    def forward(self, x: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        past = x["PAST"]
        images = x["IMAGES"]
        intent = x["INTENT"]

        if not isinstance(images, list):
            raise TypeError("Expected `IMAGES` to be a list of camera tensors.")

        assert isinstance(past, torch.Tensor)
        assert isinstance(intent, torch.Tensor)

        # Keep TFv6 backbone device in sync with distributed/lightning placement.
        self.backbone.device = past.device

        rgb = self._build_rgb_tensor(images)
        bev_features, _ = self.backbone({"rgb": rgb})
        bev_features = self.backbone.top_down(bev_features)

        bev_token = self.bev_pool(bev_features).flatten(1)
        past_token = self.past_encoder(past.reshape(past.size(0), -1))
        intent_idx = intent.long().clamp(min=0, max=self.intent_embedding.num_embeddings - 1)
        intent_token = self.intent_embedding(intent_idx)

        fused = torch.cat([bev_token, past_token, intent_token], dim=1)
        return self.head(fused)