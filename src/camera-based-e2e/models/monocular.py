from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt, log

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

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # Create a range of frequencies
        emb = log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Apply sin and cos to the timestep scaled by frequencies
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

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

class DeepMonocularModelOrig(nn.Module):
    def __init__(self, feature_extractor, out_dim, n_layers=1):
        super().__init__()
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)
        
        # Initial Query Projection (Intent + Past -> C)
        query_input_dim = 3 + 16 * 6
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
            nn.Linear(self.feature_dim, out_dim),
        )

    def forward(self, x):
        # Copied from MonocularModel
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
        
        # copy procedure to build query_0 from MonocularModel
        intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()
        past_flat = past.view(past.size(0), -1)
        query = self.query_init(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)

        for block in self.blocks:
            query = block(query, tokens)

        return self.decoder(query.squeeze(1))
    
class DeepMonocularModel(nn.Module):
    def __init__(self, feature_extractor, out_dim=2, n_layers=1):
        super().__init__()
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)

        # Initial Query Projection (Intent + Past -> C)
        query_input_dim = 3 + 16 * 6 + (2) + 20
        self.query_init = nn.Linear(query_input_dim, self.feature_dim)

        # learnable positional encoding
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)

        self.pos_proj = nn.Sequential(
            nn.Linear(2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU()
        )
        self.time_embeds = nn.Embedding(20, self.feature_dim)

        # Deep network rather than single attention in MonocularModel 
        self.blocks = nn.ModuleList([
            TransformerBlock(self.feature_dim, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 2),
        )

    def forward(self, x):
        # Copied from MonocularModel
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


        # copy procedure to build query_0 from MonocularModel
        intent_onehot = F.one_hot((intent - 1).long().clamp(0, 2), num_classes=3).float()
        past_flat = past.view(past.size(0), -1)
        context = self.query_init(torch.cat([intent_onehot, past_flat, torch.zeros(past.size(0), 22).to(past.device)], dim=1))

        outputs = []

        current_pos = past[:, -1, 0:2].clone()
        for i in range(20):            
            step_embed = self.time_embeds(torch.tensor(i).to(past.device))
            step_query = context + self.pos_proj(current_pos) + step_embed
            step_query = step_query.unsqueeze(1)

            for block in self.blocks:
                step_query = block(step_query, tokens)

            context = step_query.squeeze(1)
        
            delta_pos = self.decoder(context)
            current_pos = current_pos + delta_pos # Predict relative movement
            outputs.append(current_pos)

        return torch.stack(outputs, dim=1)

class ARMonocularModel(nn.Module):
    def __init__(self, feature_extractor, out_dim=2, n_layers=3):
        super().__init__()
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)

        # Initial Query Projection (Intent + Past -> C)
        query_input_dim = 3 + 16 * 6 + (2) + 20
        self.query_init = nn.Linear(query_input_dim, self.feature_dim)

        # learnable positional encoding
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)

        self.future_queries = nn.Parameter(torch.randn(1, 20, self.feature_dim))

        self.pos_proj = nn.Sequential(
            nn.Linear(2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU()
        )
        self.time_embeds = nn.Embedding(36, self.feature_dim) 

        self.past_encoder = nn.Linear(6, self.feature_dim)
        self.past_pos_encoder = nn.Linear(2, self.feature_dim)  # separate encoding for x,y position


        self.path_queries = nn.Parameter(torch.randn(1, 20, self.feature_dim))
        self.intent_embeds = nn.Parameter(torch.randn(1, 3,self.feature_dim))
        # Deep network rather than single attention in MonocularModel 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=16,
            dim_feedforward=self.feature_dim * 4,
            dropout=0.1,
            batch_first=True, 
            norm_first=True
        )

        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )


        self.causal_mask = None

        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 2),
        )

    def forward(self, x):
        # Copied from MonocularModel
        # past: (B, 16, 6), intent: int
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        device = past.device
        # Ref: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50%20%20order%20=%20[2,%201,%203]
        front_cam = images[1]

        with torch.no_grad():
            feats = self.features(front_cam)  # list or tensor

        # tokens: handle list of features or single tensor
        if isinstance(feats, (list, tuple)):
            image_tokens = torch.cat([f.flatten(2) for f in feats], dim=1)  # (B, C_total, N)
        else:
            image_tokens = feats.flatten(2)  # (B, C, N)
        image_tokens = torch.permute(image_tokens, (0, 2, 1)) + self.positional_encoding # (B, N, C_total)


        # copy procedure to build query_0 from MonocularModel
        intent_indices = (intent - 1).long().clamp(0, 2)  # (B,)
        intent_token = self.intent_embeds[:, intent_indices, :].squeeze(0) 
        intent_token = intent_token.unsqueeze(1)

        past_tokens = self.past_encoder(past)  # (B, 16, C)
        past_positions = self.past_pos_encoder(past[:, :, 0:2])
        past_time_embeds = self.time_embeds(torch.arange(0, 16, device=device))
        past_tokens = past_tokens + past_positions + past_time_embeds  # (B, 16, C)

        context_tokens = torch.cat([image_tokens, intent_token, past_tokens], dim=1)

        future_tokens = self.future_queries.expand(past.size(0), -1, -1)  # (B, 20, C)
        future_tokens = future_tokens + self.time_embeds(torch.arange(16, 36, device=device))

        predictions = []
    
        for t in range(20):
            current_future = future_tokens[:, :t+1, :]
            tokens = torch.cat([context_tokens, current_future], dim=1)
            
            if t == 0:
                mask = None
            else:
                mask_len = context_tokens.size(1) + t + 1
                future_start = context_tokens.size(1)
                mask = torch.zeros((mask_len, mask_len), device=device, dtype=torch.float32)
                future_mask = torch.triu(torch.ones(t+1, t+1, device=device, dtype=torch.float32), diagonal=1)
                future_mask = future_mask.masked_fill(future_mask == 1, float('-inf'))

                mask[future_start:, future_start:] = future_mask

            
            tokens = self.blocks(tokens, mask=mask)
            
            pred_t = self.decoder(tokens[:, -1:, :])  # (B, 1, 2)
            predictions.append(pred_t)
            
            if t < 19:
                future_tokens[:, t+1, :] = future_tokens[:, t+1, :] + self.pos_proj(pred_t.squeeze(1))
    
        predictions = torch.cat(predictions, dim=1)
        
        return predictions

class DiffusionMonocularModel(nn.Module):
    def __init__(self, feature_extractor, out_dim, n_layers=1):
        super().__init__()
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)
        
        # Initial Query Projection (Intent + Past -> C)
        query_input_dim = 3 + 16 * 6 + self.feature_dim + 40 
        self.query_init = nn.Linear(query_input_dim, self.feature_dim)

        # learnable positional encoding
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.SiLU(), # SiLU (Swish) is standard in diffusion
            nn.Linear(self.feature_dim * 4, self.feature_dim)
        )
        
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

    def forward(self, x, t):
        # Copied from MonocularModel
        # past: (B, 16, 6), intent: int
        past, images, intent, future = x['PAST'], x['IMAGES'], x['INTENT'], x['FUTURE']
        
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
        
        t_emb = self.time_embed(t).unsqueeze(1)
        # copy procedure to build query_0 from MonocularModel
        intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()
        past_flat = past.view(past.size(0), -1)
        query = self.query_init(torch.cat([future.view(future.size(0), -1), intent_onehot, past_flat, t_emb], dim=1)).unsqueeze(1)

        for block in self.blocks:
            query = block(query, tokens)

        return self.decoder(query.squeeze(1))
