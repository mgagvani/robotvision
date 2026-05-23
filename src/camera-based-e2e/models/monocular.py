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
        past, images, intent = x['PAST'], x['IMAGES_JPEG'], x['INTENT']
        
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
    def __init__(
        self,
        feature_extractor,
        out_dim,
        n_blocks=1,
        n_proposals=50,
        dt: float = 0.25,
        max_accel: float = 8.0,
        max_omega: float = 1.0,
    ):
        super().__init__()
        self.features = feature_extractor
        self.feature_dim = sum(self.features.dims)
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even for (x,y) rollout, got {out_dim}")
        self.horizon = out_dim // 2
        self.dt = dt
        self.max_accel = max_accel
        self.max_omega = max_omega
        
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

    def bicycle_model(self, control_pred: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        accel = torch.tanh(control_pred[..., 0]) * self.max_accel  # (B, K, T)
        omega = torch.tanh(control_pred[..., 1]) * self.max_omega  # (B, K, T)

        x_state = past[:, -1, 0].unsqueeze(1).expand(-1, self.n_proposals).clone()
        y_state = past[:, -1, 1].unsqueeze(1).expand(-1, self.n_proposals).clone()
        vx0 = past[:, -1, 2]
        vy0 = past[:, -1, 3]
        speed_state = torch.sqrt(vx0 * vx0 + vy0 * vy0 + 1e-6).unsqueeze(1).expand(-1, self.n_proposals).clone()
        heading_state = torch.atan2(vy0, vx0).unsqueeze(1).expand(-1, self.n_proposals).clone()

        xy_steps = []
        for t in range(self.horizon):
            x_state = x_state + speed_state * torch.cos(heading_state) * self.dt
            y_state = y_state + speed_state * torch.sin(heading_state) * self.dt
            xy_steps.append(torch.stack([x_state, y_state], dim=-1))

            heading_state = heading_state + omega[:, :, t] * self.dt
            speed_state = torch.clamp_min(speed_state + accel[:, :, t] * self.dt, 0.0)

        traj_xy = torch.stack(xy_steps, dim=2)  # (B, K, T, 2)
        return traj_xy, traj_xy.reshape(traj_xy.size(0), -1), accel, omega  # (B, K*T*2)

    def forward(self, x, sae=None, sae_target: str = "query", sae_latent_edit=None, sae_block_index: int | None = None):
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
        query: torch.Tensor = self.query_init(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)

        planner_query_tokens = []
        for block_idx, block in enumerate(self.blocks):
            query = block(query, tokens)
            planner_query_tokens.append(query.squeeze(1))

            if sae is not None and sae_target == "planner_query_block" and sae_block_index == block_idx:
                sae_out = sae(query.squeeze(1))
                query_latents = sae_out["latents"]
                if sae_latent_edit is not None:
                    query_latents = sae_latent_edit(query_latents)
                    query = sae.decode(query_latents).unsqueeze(1)
                else:
                    query = sae_out["reconstruction"].unsqueeze(1)
                planner_query_tokens[-1] = query.squeeze(1)

        if sae_target not in {"query", "legacy_query", "score", "traj", "control", "planner_query_block"}:
            raise ValueError(f"Unsupported SAE target '{sae_target}'")

        if sae is not None and sae_target in {"query", "legacy_query"}:
            sae_out = sae(query.squeeze(1))
            query = sae_out["reconstruction"].unsqueeze(1)

        # predict (acceleration, angular velocity) for each timestep
        # and roll it out using the kinematic bicycle model
        control_pred = self.traj_decoder(query.squeeze(1)).view(
            query.size(0), self.n_proposals, self.horizon, 2
        )  # (B, K, T, 2)

        if sae is not None and sae_target == "control":
            flat_control = control_pred.reshape(query.size(0) * self.n_proposals, self.horizon * 2)
            sae_out = sae(flat_control)
            control_latents = sae_out["latents"]
            if sae_latent_edit is not None:
                control_latents = sae_latent_edit(control_latents)
                flat_control = sae.decode(control_latents)
            else:
                flat_control = sae_out["reconstruction"]
            control_pred = flat_control.view(query.size(0), self.n_proposals, self.horizon, 2)

        traj_xy, traj_pred, accel, omega = self.bicycle_model(control_pred, past)  # (B, K, T*2)

        traj_pred_flat = traj_xy.reshape(traj_xy.size(0), self.n_proposals, -1)  # (B, K, T*2)
        traj_feat: torch.Tensor = self.traj_features(traj_pred_flat.detach())  # (B, K, C)

        if sae is not None and sae_target == "traj":
            traj_feat = sae(
                traj_feat.reshape(query.size(0) * self.n_proposals, self.feature_dim)
            )["reconstruction"].view(query.size(0), self.n_proposals, self.feature_dim)

        query_for_score = query.squeeze(1).detach()[:, torch.newaxis, :].expand(-1, self.n_proposals, -1)  # (B, K, C)
        score_in = torch.cat([query_for_score, traj_feat], dim=-1)  # (B, K, 2C)

        if sae is not None and sae_target == "score":
            score_in = sae(
                score_in.reshape(query.size(0) * self.n_proposals, self.feature_dim * 2)
            )["reconstruction"].view(query.size(0), self.n_proposals, self.feature_dim * 2)

        score_pred = self.score_decoder(score_in).squeeze(-1)  # (B, K)

        return {
            "trajectory_predicted": traj_pred,
            "trajectory_feat": traj_feat,
            "scores_predicted": score_pred,
            "scores_input": score_in,
            "depth": output_depth,
            "controls": torch.stack([accel, omega], dim=-1).reshape(query.size(0), -1),
            "query": query,
            "control_pred": control_pred,
            **{f"planner_query_tok_block_{idx}": block_query for idx, block_query in enumerate(planner_query_tokens)},
        }
