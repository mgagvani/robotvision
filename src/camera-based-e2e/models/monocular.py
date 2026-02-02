import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from .blocks import TransformerBlock

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
    def __init__(self, feature_extractor, out_dim, n_blocks=1, n_proposals=50):
        super().__init__()
        self.features = feature_extractor
        self.feature_dim = sum(self.features.dims)
        
        # Initial Query Projection (Intent + Past -> C)
        query_input_dim = 3 + 16 * 6
        self.query_init = nn.Linear(query_input_dim, self.feature_dim)

        # Instead of fine-tuning feature extractor, project w/ conv
        self.visual_adapter = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
        )

        # learnable positional encoding
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)
        
        # Deep network rather than single attention in MonocularModel 
        self.blocks = nn.ModuleList([
            TransformerBlock(self.feature_dim, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(n_blocks)
        ])

        # For Supervised Depth Loss -> (B, 128, 128)
        self.depth_gen = nn.Sequential(
            nn.Conv2d(self.feature_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 1)
        )
        
        self.n_proposals = n_proposals
        self.traj_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, out_dim * self.n_proposals),
        )
        self.traj_features = nn.Sequential(
            nn.Linear(out_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
        )
        self.score_decoder = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        ) # no softmax, since we use cross entropy later

    def forward(self, x):
        # Copied from MonocularModel
        # past: (B, 16, 6), intent: int
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        
        # Ref: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50%20%20order%20=%20[2,%201,%203]
        front_cam = images[1]

        # Doesn't need no_grad b/c DINO/SAMFeatures will freeze if needed
        feats_vit = self.features(front_cam)  # list or tensor

        if len(feats_vit) == 1 and isinstance(feats_vit, list):
            feats_vit = feats_vit[0]

        feats = self.visual_adapter(feats_vit)  # (B, C, H, W)

        # Depth Supervision
        output_depth = F.softplus(self.depth_gen(feats).squeeze(1))  # (B, 128, 128)

        # tokens: handle list of features or single tensor
        # TODO: is this made redundant by if statement above?
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

        traj_pred = self.traj_decoder(query.squeeze(1))  # (B, K*T*2)
        traj_pred_flat = traj_pred.view(traj_pred.size(0), self.n_proposals, -1)  # (B, K, T*2)
        traj_feat = self.traj_features(traj_pred_flat)  # (B, K, C)
        query_for_score = query.squeeze(1)[:, torch.newaxis, :].expand(-1, self.n_proposals, -1)  # (B, K, C)
        score_in = torch.cat([query_for_score, traj_feat], dim=-1)  # (B, K, 2C)
        score_pred = self.score_decoder(score_in).squeeze(-1)  # (B, K)

        return {
            "trajectory": traj_pred,
            "scores": score_pred,
            "depth": output_depth
        }
