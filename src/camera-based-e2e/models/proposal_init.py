"""
Proposal initialization (iPad-style).

Per-timestep learnable embeddings for N proposals × T timesteps,
conditioned on ego status (past trajectory + intent).

Output: bev_feature (B, N*T, C) — the initial BEV proposal queries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalInit(nn.Module):
    """
    Initialize per-timestep BEV proposal features from learnable embeddings
    plus ego status encoding.  Matches iPad's init_feature + hist_encoding.
    """

    def __init__(
        self,
        d_model: int,
        num_proposals: int = 16,
        horizon: int = 20,
        past_dim: int = 16 * 6,
        intent_classes: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_proposals = num_proposals
        self.horizon = horizon

        # Per-(proposal, timestep) learnable embeddings
        self.proposal_embed = nn.Parameter(
            torch.zeros(1, num_proposals * horizon, d_model)
        )
        nn.init.trunc_normal_(self.proposal_embed, std=0.1)

        ego_dim = past_dim + intent_classes
        self.ego_enc = nn.Sequential(
            nn.Linear(ego_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, past: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past: (B, 16, 6)
            intent: (B,) integer 1/2/3
        Returns:
            bev_feature: (B, N*T, d_model)
        """
        B = past.size(0)
        past_flat = past.view(B, -1)
        intent_onehot = F.one_hot(
            (intent - 1).long().clamp(0, 2), num_classes=3
        ).float()
        ego = torch.cat([intent_onehot, past_flat], dim=1)
        ego_feat = self.ego_enc(ego)  # (B, d_model)

        bev_feature = self.proposal_embed.expand(B, -1, -1) + ego_feat[:, None, :]
        return bev_feature
