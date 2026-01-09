from typing import List, Optional
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt

# Add parent directory to path to import original models
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Note: We don't import SAMFeatures/DINOFeatures here
# They will be imported and passed to this model from train_ablation.py
# This avoids namespace conflicts between:
#   /home/.../camera-based-e2e/models (top-level models package)
#   /home/.../camera-based-e2e/experiments/ablation/models (local ablation models)

class AblationMonocularModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feature_extractor: nn.Module,
        use_intent: bool = True,
    ):
        # out_dim: (B, 40) which gets reshaped to (B, 20, 2) later
        super(AblationMonocularModel, self).__init__()
        self.features = feature_extractor
        self.use_intent = use_intent

        # attention 
        self.feature_dim = sum(self.features.dims)  # works for both DINO and SAM
        self.key_projection = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim) # project into "key" space
        self.value_projection = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim)

        # condition the query on intent (B,) and past (B, 16, num_features)
        # If intent is used, add 3 for one-hot encoding, otherwise 0
        intent_dim = 3 if use_intent else 0
        query_input_dim = intent_dim + in_dim  # one hot (if used) -- concat -- flattened past
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
        # past: (B, 16, num_features), intent: int
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
        key = self.key_projection(tokens) # (B, N, feature_dim)
        value = self.value_projection(tokens) # (B, N, feature_dim)

        # Build query input
        query_inputs = []
        if self.use_intent:
            intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()  # (B, 3). minus 1 --> 0, 1, 2
            query_inputs.append(intent_onehot)
        
        past_flat = past.view(past.size(0), -1)  # (B, in_dim)
        query_inputs.append(past_flat)
        
        query = self.query(torch.cat(query_inputs, dim=1) if query_inputs else past_flat).unsqueeze(1)  # (B, 1, feature_dim)

        scores = query @ key.permute((0, 2, 1)) # (B, 1, N) single value per token
        attention = F.softmax(scores / sqrt(key.shape[2]), dim=2) @ value # (B, 1, feature_dim)
        return self.decoder(attention.squeeze(1))  # (B, out_dim)


