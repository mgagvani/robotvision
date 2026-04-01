"""
Iterative refinement (iPad-style predict-attend-refine loop).

A single shared RefinementBlock is applied K times.  At each iteration:
  1. Decode per-timestep features into full trajectory proposals
  2. Encode proposals back into feature space (proposal-anchored)
  3. Cross-attend proposal features to scene tokens
  4. FFN update
Returns all intermediate proposals for discounted supervision.
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import MHA


class RefinementBlock(nn.Module):
    """One iteration: decode proposals, encode them back, cross-attend to scene."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_proposals: int = 16,
        horizon: int = 20,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.horizon = horizon

        # Per-timestep feature → (x, y)
        self.traj_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, 2),
        )

        # Encode (x, y) back into feature space
        self.traj_enc = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.cross_attn = MHA(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(
        self,
        bev_feature: torch.Tensor,
        scene_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bev_feature: (B, N*T, C) per-timestep proposal features
            scene_feat:  (B, S, C) visual tokens from scene encoder
        Returns:
            proposals:   (B, N, T, 2) fully predicted trajectories
            bev_feature: (B, N*T, C) refined features
        """
        B = bev_feature.size(0)
        N, T = self.num_proposals, self.horizon

        proposals = self.traj_decoder(bev_feature).view(B, N, T, 2)

        prop_enc = self.traj_enc(proposals.view(B, N * T, 2))
        bev_feature = bev_feature + prop_enc

        bev_feature = bev_feature + self.cross_attn(
            self.ln1(bev_feature), context=scene_feat
        )
        bev_feature = bev_feature + self.mlp(self.ln2(bev_feature))

        return proposals, bev_feature


class Refinement(nn.Module):
    """Weight-shared iterative refinement applied num_steps times."""

    def __init__(
        self,
        d_model: int,
        num_steps: int = 4,
        num_heads: int = 8,
        num_proposals: int = 16,
        horizon: int = 20,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.block = RefinementBlock(
            d_model=d_model,
            num_heads=num_heads,
            num_proposals=num_proposals,
            horizon=horizon,
        )

    def forward(
        self,
        bev_feature: torch.Tensor,
        scene_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            proposals:     (B, N, T, 2) final iteration proposals
            bev_feature:   (B, N*T, C) final features
            proposal_list: list of (B, N, T, 2) from each iteration
        """
        proposal_list: List[torch.Tensor] = []
        proposals = None
        for _ in range(self.num_steps):
            proposals, bev_feature = self.block(bev_feature, scene_feat)
            proposal_list.append(proposals)
        return proposals, bev_feature, proposal_list
