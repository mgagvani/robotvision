"""
Top-level VLM: vision encoder -> adapter -> [vision | context | query tokens] -> decoder -> output head.
"""

import torch
import torch.nn as nn

from .adapter import Adapter
from .decoder import DecoderBase
from .output_head import NUM_TRAJECTORY_QUERIES, OutputHeadBase
from .vision_encoder import VisionEncoder


class VLMTrajectoryModel(nn.Module):
    """
    vision_encoder -> adapter -> [vision_emb | context_emb | query_emb]
    -> decoder -> output at query positions -> output_head -> (B, 40).
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        adapter: Adapter,
        decoder: DecoderBase,
        output_head: OutputHeadBase,
        num_trajectory_queries: int = NUM_TRAJECTORY_QUERIES,
    ):
        super().__init__()
        if adapter.llm_embed_dim != decoder.embed_dim:
            raise ValueError("adapter.llm_embed_dim must match decoder.embed_dim")
        self.vision_encoder = vision_encoder
        self.adapter = adapter
        self.decoder = decoder
        self.output_head = output_head
        self.num_trajectory_queries = num_trajectory_queries
        llm_dim = decoder.embed_dim
        self.query_embeddings = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(1, num_trajectory_queries, llm_dim), std=0.02)
        )

    def forward(self, x: dict) -> torch.Tensor:
        """
        x: dict with PAST (B, 16, 6), IMAGES (list of 6 tensors), INTENT (B,).
        Returns: (B, 40) trajectory (flattened 20 x 2).
        """
        past = x["PAST"]
        images = x["IMAGES"]
        intent = x["INTENT"]
        front_cam = images[1]

        vision_tokens = self.vision_encoder(front_cam)
        vision_emb, context_emb = self.adapter(vision_tokens, intent, past)
        B = vision_emb.size(0)
        query_emb = self.query_embeddings.expand(B, -1, -1)
        sequence = torch.cat([vision_emb, context_emb, query_emb], dim=1)
        hidden = self.decoder(sequence)
        start = sequence.size(1) - self.num_trajectory_queries
        hidden_at_queries = hidden[:, start:, :]
        return self.output_head(hidden_at_queries)
