"""
Scorer (paper-faithful): max-pool per-timestep BEV features -> MLP -> score.
"""
import torch
import torch.nn as nn


class Scorer(nn.Module):
    """
    Score each proposal from BEV proposal features.
    Output (B, K) raw logits — higher = better.
    """

    def __init__(
        self,
        d_model: int,
        num_proposals: int = 16,
        horizon: int = 20,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.horizon = horizon
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bev_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_feature: (B, N*T, C) per-timestep features
        Returns:
            scores: (B, N) raw logits, higher is better
        """
        B = bev_feature.size(0)
        N, T = self.num_proposals, self.horizon
        feat = bev_feature.view(B, N, T, -1).amax(dim=2)  # (B, N, C)
        scores = self.score_mlp(feat).squeeze(-1)  # (B, N)
        return scores
