from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt

from .base_model import BaseModel, LitModel
from .blocks import TransformerBlock

class DINOFeatures(nn.Module):
    def __init__(self, model_name: str = "vit_small_plus_patch16_dinov3.lvd1689m", frozen: bool = True):
        super(DINOFeatures, self).__init__()

        self.dino_model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.data_config = timm.data.resolve_data_config(model=self.dino_model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        if frozen:
            for param in self.dino_model.parameters():
                param.requires_grad = False

        self.dims = [384, 384, 384]  # feature dims for each layer
        self.patch_size = 16  # patch size

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        # transforms: resize 256x256, center crop, normalize
        x_t = self.transforms(x.float()) # preprocess
        features = self.dino_model(x_t)
        return features # 3 x [B, 384, 16, 16]
    
class SAMFeatures(nn.Module):
    def __init__(
        self,
        model_name: str = "timm/sam2_hiera_tiny.fb_r896_2pt1",
        frozen: bool = True,
        feature_stage: int = -1,
    ):
        super(SAMFeatures, self).__init__()

        # features_only returns a list of stage outputs
        self.sam_model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.data_config = timm.data.resolve_data_config(model=self.sam_model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        if frozen:
            for param in self.sam_model.parameters():
                param.requires_grad = False

        channels = self.sam_model.feature_info.channels()
        reductions = self.sam_model.feature_info.reduction()
        self.feature_stage = feature_stage
        self.dims = [channels[feature_stage]]
        self.patch_size = reductions[feature_stage]  # effective stride

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        x_t = self.transforms(x.float())  # preprocess
        feats = self.sam_model(x_t)       # list of feature maps
        return [feats[self.feature_stage]]

class MonocularModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feature_extractor: nn.Module
    ):
        # out_dim: (B, 40) which gets reshaped to (B, 20, 2) later
        super(MonocularModel, self).__init__()
        self.features = feature_extractor

        # attention 
        self.feature_dim = sum(self.features.dims)  # works for both DINO and SAM
        self.key_projection = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim) # project into "key" space
        self.value_projection = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim)

        # condition the query on intent (B,) and past (B, 16, 6)
        query_input_dim = 3 + 16 * 6  # one hot -- concat -- flattened
        self.query = nn.Sequential(
            nn.Linear(query_input_dim, self.feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # learnable positional encoding
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)

        # MLP at end rather than directly using softmax as final output
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, out_dim),
        )

        # LayerNorms
        self.token_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attn_norm = nn.LayerNorm(self.feature_dim)


    def forward(self, x: dict) -> torch.Tensor:
        # past: (B, 16, 6), intent: int
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        
        # Ref: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50%20%20order%20=%20[2,%201,%203]
        front_cam = images[1]
        with torch.no_grad():
            feats = self.features(front_cam)  # list or tensor

        # tokens: handle list of features or single tensor
        if isinstance(feats, (list, tuple)):
            tokens = torch.cat([f.flatten(2) for f in feats], dim=1)  # (B, C_total, N)
        else:
            tokens = feats.flatten(2)  # (B, C, N)
        tokens = torch.permute(tokens, (0, 2, 1)) + self.positional_encoding # (B, N, C_total)
        tokens = self.token_norm(tokens)

        # attention
        key = self.key_projection(tokens) # (B, 256, 1152)
        value = self.value_projection(tokens) # (B, 256, 40)

        intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()  # (B, 3). minus 1 --> 0, 1, 2
        past_flat = past.view(past.size(0), -1)  # (B, 96)
        query = self.query(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)  # (B, 1, 256)
        query = self.query_norm(query)

        scores = query @ key.permute((0, 2, 1)) # (B, T, N)
        attention = F.softmax(scores / sqrt(key.shape[2]), dim=2) @ value # (B, 1, 40)
        attention = self.attn_norm(attention)
        return self.decoder(attention.squeeze(1))  # (B, 40)

class DeepMonocularModel(nn.Module):
    def __init__(self, feature_extractor, out_dim=2, n_layers=1):
        super().__init__()
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)
        
        # Initial Query Projection (Intent + Past -> C)
        query_input_dim = 3 + 16 * 6 + (2 * 20)
        self.query_init = nn.Linear(query_input_dim, self.feature_dim)

        # learnable positional encoding
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)
        
        # Deep network rather than single attention in MonocularModel 
        self.blocks = nn.ModuleList([
            TransformerBlock(self.feature_dim, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(n_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 40),
        )

    def forward(self, x):
        # Copied from MonocularModel
        # past: (B, 16, 6), intent: int
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        B=past.size(0)
        
        # Ref: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50%20%20order%20=%20[2,%201,%203]
        front_cam = images[1]

        with torch.no_grad():
            feats = self.features(front_cam)  # list or tensor
        

        # tokens: handle list of features or single tensor
        if isinstance(feats, (list, tuple)):
            tokens = torch.cat([f.flatten(2) for f in feats], dim=1)  # (B, C_total, N)
        else:
            tokens = feats.flatten(2)  # (B, C, N)
        tokens = torch.permute(tokens, (0, 2, 1)) + self.positional_encoding # (B, N, C_total)
        
        # copy procedure to build query_0 from MonocularModel
        intent_onehot = F.one_hot((intent - 1).long().clamp(0, 2), num_classes=3).float()
        past_flat = past.view(past.size(0), -1)
        base_inputs = torch.cat([intent_onehot, past_flat], dim=1)
     

        tradj = torch.randn((B, 20, 2), device=past.device)

        for _ in range(20):

            tradj_flat = tradj.reshape(B, -1)
            query_input = torch.cat([base_inputs, tradj_flat], dim=1)
            
            query = self.query_init(query_input).unsqueeze(1)
            
            for block in self.blocks:
                query = block(query, tokens)

            # Update tradj: Decoder outputs (B, 40), we reshape to (B, 20, 2)
            tradj = self.decoder(query.squeeze(1)).view(B, 20, 2)
            

        return tradj