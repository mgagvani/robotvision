from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt, log

from .base_model import BaseModel, LitModel
from .blocks import TransformerBlock
from .vae import LSTM_VAE, VAEModel

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
        x_t = self.transforms(x.float().div(255.0)) # preprocess
        features = self.dino_model(x_t)
        return features # 3 x [B, 384, 16, 16]

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Handle different input shapes
        original_shape = time.shape
        time = time.flatten()  # Flatten to 1D
        
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Restore original batch dimensions if needed
        if len(original_shape) > 1:
            embeddings = embeddings.view(*original_shape, -1)
        
        return embeddings

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
        x_t = self.transforms(x.float().div(255.0))  # preprocess
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

class DeepMonocularModelVAE(nn.Module):
    def __init__(self, feature_extractor, out_dim, n_layers=1):
        super().__init__()
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)
        self.vae = VAEModel.load_from_checkpoint("/home/bnamikas/git/robotvision/src/camera-based-e2e/e2e-vae-epoch=00-val_loss=3.88.ckpt", model=LSTM_VAE())
        self.vae.model.eval()
        for param in self.vae.model.parameters():
            param.requires_grad = False

        
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
            nn.Sigmoid(),
            nn.Linear(self.feature_dim, self.vae.model.latent_dim)
        )

    def forward(self, x, *args, **kwargs):
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

        latent = self.decoder(query.squeeze(1))

        z_repeated = latent.unsqueeze(1).repeat(1, 20, 1)
        decoder_out, _ = self.vae.model.decoder_lstm(z_repeated)
        output = self.vae.model.final_layer(decoder_out)
        
        return output


class DeepMonocularModelAR(nn.Module):
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
    def __init__(self, feature_extractor, out_dim=0, n_layers=3, n_heads=16):
        super().__init__()
        self.features = feature_extractor
        # Freeze backbone for memory efficiency
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.feature_dim = sum(self.features.dims)
        
        # 1. Image Spatial Encoding
        # Assuming input 224x224 and patch 16 -> 14x14 = 196 tokens
        self.n_img_tokens = 1024
        self.img_pos_enc = nn.Parameter(torch.randn(1, self.n_img_tokens, self.feature_dim) * 0.02)
        
        # 2. Temporal & Metadata Encodings
        self.time_embeds = nn.Embedding(36, self.feature_dim) # 0-15 past, 16-35 future
        self.intent_embeds = nn.Embedding(3, self.feature_dim)
        self.past_encoder = nn.Linear(6, self.feature_dim)
        
        # 3. Trajectory Projection (The AR "Bridge")
        self.pos_proj = nn.Linear(2, self.feature_dim)
        
        # 4. Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=n_heads,
            dim_feedforward=self.feature_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. Output Head
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 2) # Predicting (x, y)
        )

    def _get_image_features(self, images):
        with torch.no_grad():
            self.features.eval()
            # Front camera is usually index 1 in your setup
            feats = self.features(images[1]) 
            if isinstance(feats, (list, tuple)):
                feats = torch.cat([f.flatten(2) for f in feats], dim=1)
            else:
                feats = feats.flatten(2)
        return feats.permute(0, 2, 1) + self.img_pos_enc

    def forward(self, x, stage, noise_std=0.05):
        past, images, intent, target_traj = x['PAST'], x['IMAGES'], x['INTENT'], x["FUTURE"]
        device = past.device
        B = past.shape[0]

        # --- Context Preparation ---
        img_tokens = self._get_image_features(images)
        
        intent_idx = (intent - 1).long().clamp(0, 2)
        intent_tok = self.intent_embeds(intent_idx).unsqueeze(1)
        
        past_toks = self.past_encoder(past) + self.time_embeds(torch.arange(16, device=device))
        
        context = torch.cat([img_tokens, intent_tok, past_toks], dim=1)
        
        # --- Training Mode (Parallel with Teacher Forcing + Noise) ---
        if stage != "val" and target_traj is not None:
            # Shift target to use as input: [Last Past Pos, T0, T1... T18]
            last_pos = past[:, -1:, :2]
            tf_inputs = torch.cat([last_pos, target_traj[:, :-1, :2]], dim=1)
            
            # Injection of noise to force error correction learning
            tf_inputs = tf_inputs + torch.randn_like(tf_inputs) * noise_std
            
            future_toks = self.pos_proj(tf_inputs) + self.time_embeds(torch.arange(16, 36, device=device))
            
            full_seq = torch.cat([context, future_toks], dim=1)
            
            # Causal mask for the future tokens
            seq_len = full_seq.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            
            out = self.blocks(full_seq, mask=mask)
            # Only decode the last 20 tokens (the future)
            return self.decoder(out[:, -20:, :])

        # --- Inference Mode (Autoregressive Loop) ---
        else:
            current_seq = context
            predictions = []
            last_coord = past[:, -1:, :2]
            
            for t in range(20):
                # Prepare mask for current sequence length
                curr_len = current_seq.size(1)
                mask = torch.triu(torch.ones(curr_len, curr_len, device=device), diagonal=1).bool()
                
                # Only need the last output to predict the next step
                out = self.blocks(current_seq, mask=mask)
                pred_t = self.decoder(out[:, -1:, :])
                predictions.append(pred_t)
                
                # Feedback loop
                next_tok = self.pos_proj(pred_t) + self.time_embeds(torch.tensor([16 + t], device=device))
                current_seq = torch.cat([current_seq, next_tok], dim=1)
                
            return torch.cat(predictions, dim=1)
