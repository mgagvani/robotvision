import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .blocks import TransformerBlock

@dataclass
class DeepMonocularConfig:
    # arch
    n_blocks: int = 4
    n_proposals: int = 50
    cam_idxs_used: tuple = (1,) # front only
    # kinematics
    dt: float = 0.25
    max_accel: float = 8.0
    max_omega: float = 1.0
    # training
    use_depth_loss: bool = False
    # --- PR #16 (Scorer as Optimization Target) ---
    # https://github.com/mgagvani/robotvision/pull/16
    # adversarial training
    adv_enabled: bool = False
    adv_lambda: float = 0.1
    adv_epsilon: float = 0.10
    adv_steps: int = 3
    # sobolev training
    sobolev_enabled: bool = True
    sobolev_lambda: float = 500 # scale up loss further

class DeepMonocularModel(nn.Module):
    def __init__(
        self,
        feature_extractor,
        out_dim,
    ):
        super().__init__()
        self.cfg = DeepMonocularConfig()
        self.features = feature_extractor
        self.feature_dim = sum(self.features.dims)
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even for (x,y) rollout, got {out_dim}")
        self.horizon = out_dim // 2
        self.dt = self.cfg.dt
        self.max_accel = self.cfg.max_accel
        self.max_omega = self.cfg.max_omega
        
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
        if hasattr(self.features, "n_tokens"):
            self.n_tokens = self.features.n_tokens
        else:
            self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)
        
        # Deep network rather than single attention in MonocularModel 
        self.blocks = nn.ModuleList([
            TransformerBlock(self.feature_dim, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(self.cfg.n_blocks)
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
        
        self.n_proposals = self.cfg.n_proposals
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

    def rollout_controls(
        self,
        controls: torch.Tensor,
        past: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if controls.ndim != 4 or controls.size(-1) != 2:
            raise ValueError(
                f"controls must have shape (B, K, T, 2), got {tuple(controls.shape)}"
            )

        batch_size, n_proposals, horizon, _ = controls.shape
        accel = controls[..., 0]
        omega = controls[..., 1]

        x_state = past[:, -1, 0].unsqueeze(1).expand(-1, n_proposals).clone()
        y_state = past[:, -1, 1].unsqueeze(1).expand(-1, n_proposals).clone()
        vx0 = past[:, -1, 2]
        vy0 = past[:, -1, 3]
        speed_state = torch.sqrt(vx0 * vx0 + vy0 * vy0 + 1e-6).unsqueeze(1).expand(-1, n_proposals).clone()
        heading_state = torch.atan2(vy0, vx0).unsqueeze(1).expand(-1, n_proposals).clone()

        xy_steps = []
        for t in range(horizon):
            x_state = x_state + speed_state * torch.cos(heading_state) * self.dt
            y_state = y_state + speed_state * torch.sin(heading_state) * self.dt
            xy_steps.append(torch.stack([x_state, y_state], dim=-1))

            heading_state = heading_state + omega[:, :, t] * self.dt
            speed_state = torch.clamp_min(speed_state + accel[:, :, t] * self.dt, 0.0)

        traj_xy = torch.stack(xy_steps, dim=2)  # (B, K, T, 2)
        return traj_xy, traj_xy.reshape(batch_size, n_proposals, -1)

    def bicycle_model(
        self,
        control_pred: torch.Tensor,
        past: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        accel = torch.tanh(control_pred[..., 0]) * self.max_accel  # (B, K, T)
        omega = torch.tanh(control_pred[..., 1]) * self.max_omega  # (B, K, T)
        controls = torch.stack([accel, omega], dim=-1)
        traj_xy, traj_flat = self.rollout_controls(controls, past)
        return traj_xy, traj_flat, accel, omega

    def score_trajectories(
        self,
        traj_flat: torch.Tensor,
        query_for_score: torch.Tensor,
    ) -> torch.Tensor:
        traj_feat = self.traj_features(traj_flat)
        score_in = torch.cat([query_for_score.to(dtype=traj_feat.dtype), traj_feat], dim=-1) # (B, K, C*2)
        return self.score_decoder(score_in).squeeze(-1) # (B, K)

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
        query: torch.Tensor = self.query_init(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)

        for block in self.blocks:
            query = block(query, tokens)

        # predict (acceleration, angular velocity) for each timestep
        # and roll it out using the kinematic bicycle model
        control_pred = self.traj_decoder(query.squeeze(1)).view(
            query.size(0), self.n_proposals, self.horizon, 2
        )  # (B, K, T, 2)
        traj_xy, traj_pred_flat, accel, omega = self.bicycle_model(control_pred, past)  # (B, K, T*2)

        query_for_score = query.squeeze(1).detach()[:, torch.newaxis, :].expand(-1, self.n_proposals, -1)  # (B, K, C)
        score_pred = self.score_trajectories(traj_pred_flat.detach(), query_for_score)  # (B, K)
        controls_structured = torch.stack([accel, omega], dim=-1)

        return {
            "trajectory": traj_pred_flat.reshape(traj_pred_flat.size(0), -1),
            "trajectory_flat": traj_pred_flat,
            "query_for_score": query_for_score,
            "scores": score_pred,
            "depth": output_depth,
            "controls": controls_structured.reshape(query.size(0), -1),
            "controls_structured": controls_structured,
        }
