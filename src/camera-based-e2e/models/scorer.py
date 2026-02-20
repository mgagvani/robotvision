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


class ScorerModel(nn.Module):
    """
    Multi-proposal trajectory generator + proposal scorer.
    """

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

        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even for (x,y) rollout, got {out_dim}")
        if self.cfg.state_size != 2:
            raise ValueError(f"ScorerModel expects state_size=2 for (x, y), got {self.cfg.state_size}")
        self.horizon = out_dim // 2

        h, w = self.features.data_config["input_size"][1:]
        self.n_img_tokens = (h // self.features.patch_size) * (w // self.features.patch_size)

        # Perception encoder: image tokens + learnable scene registers from DrivoR
        self.scene_embeds = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((1, self.cfg.num_scene_tokens, self.d_features)), std=0.02),
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

    def _extract_tokens(self, front_cam: torch.Tensor) -> torch.Tensor:
        '''
        separate function to flatten vit features
        '''
        feats_vit = self.features(front_cam)  # list/tuple or tensor
        if isinstance(feats_vit, (list, tuple)):
            feats_vit = torch.cat([f.flatten(2) for f in feats_vit], dim=1)  # (B, C_total, N)
        else:
            feats_vit = feats_vit.flatten(2)  # (B, C, N)
        return feats_vit.permute(0, 2, 1)  # (B, N, d_features)

    def forward(self, x):
        # past: (B, 16, 6), intent: int
        past, images, intent = x["PAST"], x["IMAGES"], x["INTENT"]
        B = past.size(0)

        front_cam = images[1]  # (B, 3, H, W)

        # perception encoding
        tokens = self._extract_tokens(front_cam)  # (B, N, d_features)
        scene_tokens = self.scene_embeds.expand(B, -1, -1)  # (B, S, d_features)
        pe_inputs = torch.cat([tokens, scene_tokens], dim=1)  # (B, N + S, d_features)
        scene_features = self.perception_encoder(pe_inputs + self.pe_pos_embed)  # (B, N + S, d_features)
        scene_features = self.pe_proj(scene_features)  # (B, N + S, d_model)

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

        trajectory = proposals.reshape(B, -1)  # (B, K*T*2)

        return {
            "trajectory": trajectory,
            "scores": score_pred,
        }
