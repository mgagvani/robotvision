"""
Output head: map decoder hidden states at query positions to trajectory (B, 40).
Modular: base + TrajectoryHead (Option A: 20 queries -> MLP -> (B, 20, 2)).
"""

import torch
import torch.nn as nn

NUM_TRAJECTORY_QUERIES = 20
TRAJECTORY_OUTPUT_DIM = 40  # 20 * 2

class OutputHeadBase(nn.Module):
    """
    Base for output heads. Consumes (B, num_queries, embed_dim), returns (B, 40).
    """

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, num_queries, embed_dim) from decoder at query positions.
        Returns: (B, 40) trajectory (flattened 20 x 2).
        """
        raise NotImplementedError


class TrajectoryHead(OutputHeadBase):
    """
    MLP on each query position -> (x, y). Output (B, num_queries, 2), flatten -> (B, 40).
    """

    def __init__(self, embed_dim: int, num_queries: int = NUM_TRAJECTORY_QUERIES):
        super().__init__()
        self.num_queries = num_queries
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, num_queries, embed_dim)
        Returns: (B, 40)
        """
        out = self.mlp(hidden)
        return out.flatten(1)
