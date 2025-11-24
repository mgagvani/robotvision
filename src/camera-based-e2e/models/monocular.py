from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt

from .base_model import BaseModel, LitModel

class DINOFeatures(nn.Module):
    def __init__(self, model_name: str = "vit_small_plus_patch16_dinov3.lvd1689m", frozen: bool = True):
        super(DINOFeatures, self).__init__()

        self.dino_model = timm.create_model(model_name, pretrained=True, features_only=True)
        data_config = timm.data.resolve_data_config(model=self.dino_model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        if frozen:
            for param in self.dino_model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        # transforms: resize 256x256, center crop, normalize
        x_t = self.transforms(x.float()) # preprocess
        features = self.dino_model(x_t)
        return features # 3 x [B, 384, 16, 16]

class MonocularModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dino_model_name: str = "vit_small_plus_patch16_dinov3.lvd1689m"):
        # out_dim: (B, 40) which gets reshaped to (B, 20, 2) later
        super(MonocularModel, self).__init__()
        self.dino = DINOFeatures(model_name=dino_model_name, frozen=True)

        # vanilla attention 
        self.query = nn.Parameter(torch.zeros(1, 1, 1152)) # (1, 1, 1152)
        nn.init.normal_(self.query) # init to N(0, 1)
        self.key_projection = nn.Linear(in_features=1152, out_features=1152) # project (1152,) into "key" space
        self.value_projection = nn.Linear(in_features=1152, out_features=out_dim) # project (1152,) into output dimension


    def forward(self, x: dict) -> torch.Tensor:
        # past: (B, 16, 6), intent: int
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        
        # Ref: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50%20%20order%20=%20[2,%201,%203]
        front_cam = images[1]
        dino_features: List[torch.Tensor] = self.dino(front_cam)  # 3 x [B, 384, 16, 16]

        # tokens
        tokens = torch.cat([f.flatten(2) for f in dino_features], dim=1)  # (B, 384*3, 16*16) = (B, 1152, 256)
        tokens = torch.permute(tokens, (0, 2, 1)) # (B, 256, 1152)

        # attention
        key = self.key_projection(tokens) # (B, 256, 1152)
        value = self.value_projection(tokens) # (B, 256, 40)
        query = self.query.broadcast_to((tokens.shape[0], 1, 1152)) # (B, 1, 1152)

        scores = query @ key.permute((0, 2, 1)) # (B, 1, 256) single value per token
        attention = F.softmax(scores / sqrt(key.shape[2]), dim=2) @ value # (B, 1, 40)
        return attention.squeeze(1) # (B, 40)

