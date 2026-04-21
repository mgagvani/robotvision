"""
Proposal-centric E2E planner (iPad-style).

Pipeline:
  scene encoder → proposal init → iterative refinement → scorer
                                  (with intermediate proposal supervision)
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .scene_encoder import SceneEncoder
from .proposal_init import ProposalInit
from .refinement import Refinement
from .scorer import Scorer


class ProposalPlanner(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 256,
        num_proposals: int = 16,
        num_refinement_steps: int = 4,
        horizon: int = 20,
        num_cams: int = 8,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_proposals = num_proposals

        self.scene_encoder = SceneEncoder(backbone, d_model=d_model, num_cams=num_cams)
        self.proposal_init = ProposalInit(
            d_model=d_model,
            num_proposals=num_proposals,
            horizon=horizon,
        )
        self.refinement = Refinement(
            d_model=d_model,
            num_steps=num_refinement_steps,
            num_heads=8,
            num_proposals=num_proposals,
            horizon=horizon,
        )
        self.scorer = Scorer(
            d_model=d_model,
            num_proposals=num_proposals,
            horizon=horizon,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        past = x["PAST"]
        images: List[torch.Tensor] = x["IMAGES"]
        intent = x["INTENT"]

        scene_feat = self.scene_encoder(images)
        bev_feature = self.proposal_init(past, intent)
        proposals, bev_feature, proposal_list = self.refinement(bev_feature, scene_feat)
        scores = self.scorer(bev_feature)

        if getattr(self, "_debug", False):
            self._debug_proposals = proposals.detach()
            self._debug_scores = scores.detach()
            self._debug_proposal_list = [p.detach() for p in proposal_list]

        B, K, T, _ = proposals.shape
        trajectory_flat = proposals.view(B, K * T * 2)

        return {
            "trajectory": trajectory_flat,
            "scores": scores,
            "proposal_list": proposal_list,
        }
