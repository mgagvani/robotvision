import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from .blocks import MLP, TransformerDecoder, TransformerDecoderScorer


@dataclass
class ScorerConfig:
    proposal_num: int = 64
    state_size: int = 2
    num_scene_tokens: int = 16
    ref_num: int = 4
    scorer_ref_num: int = 4
    d_model: int = 256
    d_ffn: int = 2048
    pe_num_layers: int = 2
    pe_num_heads: int = 8
    pe_d_ffn: int = 4096
    n_heads: int = 8
    n_cameras: int = 3


class ScorerModel(nn.Module):
    """
    Multi-proposal trajectory generator + proposal scorer.
    """
    HORIZON = 20
    DT = 0.25
    MAX_ACCEL = 8.0
    MAX_OMEGA = 1.0

    def __init__(
        self,
        feature_extractor,
        out_dim,
        **kwargs, # accept kwargs so we can pass in random things
    ):
        super().__init__()
        self.cfg = ScorerConfig()
        self.features = feature_extractor
        self.d_features = sum(self.features.dims)
        self.n_proposals = self.cfg.proposal_num

        # F, FL, FR, R, L, R, BL, BR respectively. choose the first n_cameras
        self.cameras = [1, 2, 3, 7, 4, 5, 6, 8][: self.cfg.n_cameras] 

        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even for (x,y) rollout, got {out_dim}")
        if self.cfg.state_size != 2:
            raise ValueError(f"ScorerModel expects state_size=2 for (x, y), got {self.cfg.state_size}")
        self.horizon = out_dim // 2

        h, w = self.features.data_config["input_size"][1:]
        self.n_img_tokens = (h // self.features.patch_size) * (w // self.features.patch_size)

        # Perception encoder: image tokens + learnable scene registers from DrivoR
        self.scene_embeds = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((1, self.cfg.n_cameras, self.cfg.num_scene_tokens, self.d_features)), std=0.02),
        )
        pe_layer = nn.TransformerEncoderLayer(
            d_model=self.d_features,
            nhead=self.cfg.pe_num_heads,
            dim_feedforward=self.cfg.pe_d_ffn,
            dropout=0.1,
            activation=F.gelu,
            batch_first=True,
        )
        self.perception_encoder = nn.TransformerEncoder(
            pe_layer,
            num_layers=self.cfg.pe_num_layers,
            norm=nn.LayerNorm(self.d_features),
        )
        self.pe_pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_img_tokens + self.cfg.num_scene_tokens, self.d_features)), std=0.02))        
        self.pe_proj = nn.Linear(self.d_features, self.cfg.d_model)

        # Proposal generator
        self.history_encoding = nn.Linear(3 + 16 * 6, self.cfg.d_model)  # intent onehot + past
        self.traj_features = nn.Embedding(self.cfg.proposal_num, self.cfg.d_model)
        self.traj_decoder = TransformerDecoder(
            num_layers=self.cfg.ref_num,
            dim=self.cfg.d_model,
            num_heads=self.cfg.n_heads,
        )
        self.traj_heads = nn.ModuleList(
            [
                MLP(self.cfg.d_model, self.cfg.d_ffn, self.horizon * self.cfg.state_size)
                for _ in range(self.cfg.ref_num + 1)
            ]
        )

        # Proposal scorer
        self.scorer_decoder = TransformerDecoderScorer(
            num_layers=self.cfg.scorer_ref_num,
            dim=self.cfg.d_model,
            num_heads=self.cfg.n_heads,
        )
        self.pos_embed = MLP(self.horizon * self.cfg.state_size, self.cfg.d_ffn, self.cfg.d_model)
        self.scorer = MLP(self.cfg.d_model, self.cfg.d_ffn, 1)

    def extract_tokens(self, cameras: torch.Tensor) -> torch.Tensor:
        '''
        separate function to flatten vit features
        '''
        B, N, C, H, W = cameras.shape  # (B, n_cameras, 3, H, W)
        cam_inputs = cameras.reshape(B * N, C, H, W)  
        feats_vit = self.features(cam_inputs)  # list/tuple or tensor
        if isinstance(feats_vit, (list, tuple)):
            feats_vit = torch.cat([f.flatten(2) for f in feats_vit], dim=1)  # (B, C_total, N)
        else:
            feats_vit = feats_vit.flatten(2)  # (B, C, N)
        tokens = feats_vit.permute(0, 2, 1)  
        T = tokens.shape[1]
        return tokens.reshape(B, N, T, self.d_features)
    
    def bicycle_model(self, control_pred, past):    
        accel = torch.tanh(control_pred[..., 0]) * self.MAX_ACCEL  # (B, K, T)
        omega = torch.tanh(control_pred[..., 1]) * self.MAX_OMEGA  # (B, K, T)

        x_state = past[:, -1, 0].unsqueeze(1).expand(-1, self.n_proposals).clone()
        y_state = past[:, -1, 1].unsqueeze(1).expand(-1, self.n_proposals).clone()
        vx0 = past[:, -1, 2]
        vy0 = past[:, -1, 3]
        speed_state = torch.sqrt(vx0 * vx0 + vy0 * vy0 + 1e-8).unsqueeze(1).expand(-1, self.n_proposals).clone()
        heading_state = torch.atan2(vy0, vx0).unsqueeze(1).expand(-1, self.n_proposals).clone()

        xy_steps = []
        for t in range(self.HORIZON):
            x_state = x_state + speed_state * torch.cos(heading_state) * self.DT
            y_state = y_state + speed_state * torch.sin(heading_state) * self.DT
            xy_steps.append(torch.stack([x_state, y_state], dim=-1))

            heading_state = heading_state + omega[:, :, t] * self.DT
            speed_state = torch.clamp_min(speed_state + accel[:, :, t] * self.DT, 0.0)

        traj_xy = torch.stack(xy_steps, dim=2)  # (B, K, T, 2)
        traj_pred = traj_xy.reshape(traj_xy.size(0), -1)  # (B, K*T*2)
        return traj_pred

    def forward(self, x):
        # past: (B, 16, 6), intent: int
        past, images, intent = x["PAST"], x["IMAGES"], x["INTENT"]
        B = past.size(0)

        camera_images = torch.stack([images[i] for i in self.cameras], axis=1)  # (B, n_cameras, 3, H, W)

        # perception encoding
        tokens = self.extract_tokens(camera_images)  # (B, N, T, d_features)
        N, S = tokens.shape[1], self.cfg.num_scene_tokens
        scene_tokens = self.scene_embeds.expand(B, -1, -1, -1)  # (B, N, S, d_features)
        
        pe_inputs = torch.cat([tokens, scene_tokens], dim=2)  # (B, N, T+S, d_features)
        pe_inputs = pe_inputs + self.pe_pos_embed.unsqueeze(1)  # Add pos_embed before flattening
        pe_inputs = pe_inputs.view(B * N, -1, self.d_features)  # per camera attention. do not want attending across cameras
        
        scene_features = self.perception_encoder(pe_inputs)  # (B*N, T+S, d_features)
        scene_features = scene_features[:, -S:, :].reshape(B, N * S, self.d_features)  # Extract scene tokens
        
        scene_features = self.pe_proj(scene_features)  # (B, N*S, d_model)

        # token conditioning on intent and past states
        intent_idx = (intent.long() - 1).clamp(min=0, max=2)
        intent_onehot = F.one_hot(intent_idx, num_classes=3).float()  # (B, 3)
        past_flat = past.view(B, -1)  # (B, 96)
        ego_token = self.history_encoding(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)  # (B, 1, d_model)

        # query tokens for proposed trajectories
        traj_tokens = self.traj_features.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)
        traj_tokens = traj_tokens + ego_token  # (B, K, d_model)

        # Iterative proposal decoding
        traj_proposals = []
        proposals = self.traj_heads[0](traj_tokens).reshape(B, self.cfg.proposal_num, self.horizon, self.cfg.state_size)
        traj_proposals.append(proposals)

        decoder_states = self.traj_decoder(traj_tokens, scene_features)
        for i, state in enumerate(decoder_states):
            proposals = self.traj_heads[i + 1](state).reshape(B, self.cfg.proposal_num, self.horizon, self.cfg.state_size)
            traj_proposals.append(proposals)

        proposals = traj_proposals[-1]  # (B, K, T, 2)

        # Score each proposal and detach so score loss does not backprop into trajectory gen
        traj_flat = proposals.view(B, self.cfg.proposal_num, -1).detach()  # (B, K, T*2)
        traj_embedding = self.pos_embed(traj_flat)  # (B, K, d_model)
        scorer_out = self.scorer_decoder(traj_embedding, scene_features)  # (B, K, d_model)
        scorer_out = scorer_out + ego_token
        score_pred = self.scorer(scorer_out).squeeze(-1)  # (B, K)

        # trajectory = proposals.reshape(B, -1)  # (B, K*T*2)
        trajectory = self.bicycle_model(proposals, past)  # (B, K*T*2) use bicycle model to constrain output

        return {
            "trajectory": trajectory,
            "scores": score_pred,
        }
