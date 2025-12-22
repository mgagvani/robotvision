from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt

from .base_model import BaseModel, LitModel

class ViTIntermediateFeatures(nn.Module):
    """
    Generic ViT feature extractor that returns token maps from selected transformer blocks.
    """
    def __init__(
        self,
        model_name: str,
        layer_indices: Optional[List[int]] = None,
        frozen: bool = True,
        use_cls_token: bool = False,
    ):
        super(ViTIntermediateFeatures, self).__init__()

        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True)
        if not hasattr(self.model, "blocks"):
            raise ValueError(f"{model_name} is not a ViT-style model with transformer blocks")

        self.num_blocks = len(self.model.blocks)
        if layer_indices is None:
            start = self.num_blocks // 2
            self.layer_indices = list(range(start, self.num_blocks))
        else:
            for idx in layer_indices:
                if idx < 0 or idx >= self.num_blocks:
                    raise ValueError(f"Layer index {idx} out of range for {model_name} with {self.num_blocks} blocks")
            self.layer_indices = sorted(layer_indices)

        self.data_config = timm.data.resolve_data_config(model=self.model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        self.use_cls_token = use_cls_token
        self.frozen = frozen

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.embed_dim = getattr(self.model, "embed_dim", getattr(self.model, "num_features", None))
        self.dims = [self.embed_dim for _ in self.layer_indices]

        patch = getattr(self.model, "patch_embed", None)
        patch_size = getattr(patch, "patch_size", None) if patch is not None else None
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        if patch_size is None and patch is not None and hasattr(patch, "num_patches"):
            num_patches = getattr(patch, "num_patches")
            side = int(round(sqrt(num_patches))) if num_patches > 0 else 0
            input_h = self.data_config["input_size"][1]
            patch_size = input_h // side if side > 0 else None

        # ensure patch_size is always set to a sensible default
        self.patch_size = patch_size or 16

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert sequence tokens to (B, C, H, W) grid."""
        num_prefix = getattr(self.model, "num_prefix_tokens", 1 if getattr(self.model, "cls_token", None) is not None else 0)
        if not self.use_cls_token and num_prefix > 0:
            tokens = tokens[:, num_prefix:, :]

        B, seq_len, dim = tokens.shape
        side = int(round(sqrt(seq_len)))
        if side * side != seq_len:
            # fall back to (B, C, N, 1) when not a perfect square
            return tokens.transpose(1, 2).unsqueeze(-1)

        if self.patch_size is None and "input_size" in self.data_config:
            input_h = self.data_config["input_size"][1]
            if input_h % side == 0:
                self.patch_size = input_h // side

        return tokens.transpose(1, 2).reshape(B, dim, side, side)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_t = self.transforms(x.float())  # preprocess to model expected size

        outputs = {}
        handles = []
        for layer_idx in self.layer_indices:
            handle = self.model.blocks[layer_idx].register_forward_hook(
                lambda module, inp, out, idx=layer_idx: outputs.setdefault(idx, out)
            )
            handles.append(handle)

        forward_ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with forward_ctx:
            _ = self.model.forward_features(x_t)

        for handle in handles:
            handle.remove()

        maps = [self._tokens_to_map(outputs[idx]) for idx in self.layer_indices]
        return maps

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

        # attention
        key = self.key_projection(tokens) # (B, 256, 1152)
        value = self.value_projection(tokens) # (B, 256, 40)

        intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()  # (B, 3). minus 1 --> 0, 1, 2
        past_flat = past.view(past.size(0), -1)  # (B, 96)
        query = self.query(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)  # (B, 1, 256)


        scores = query @ key.permute((0, 2, 1)) # (B, 1, 256) single value per token
        attention = F.softmax(scores / sqrt(key.shape[2]), dim=2) @ value # (B, 1, 40)
        return self.decoder(attention.squeeze(1))  # (B, 40)
