"""
Adapter: map vision encoder output and context (intent + past) into LLM embedding space.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Waymo E2E: intent one-hot (3) + past flat (16 * 6)
CONTEXT_INPUT_DIM = 3 + 16 * 6

class Adapter(nn.Module):
    """
    Maps vision sequence and context (intent + past) into LLM embedding space.
    Vision: optional downsample -> linear proj. Context: norm -> MLP -> context tokens.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_embed_dim: int,
        num_context_tokens: int = 4,
        num_vision_tokens_after_downsample: Optional[int] = None,
        context_input_dim: int = CONTEXT_INPUT_DIM,
    ):
        super().__init__()
        self.llm_embed_dim = llm_embed_dim
        self.num_context_tokens = num_context_tokens
        self.num_vision_tokens_after_downsample = num_vision_tokens_after_downsample

        self.vision_proj = nn.Linear(vision_dim, llm_embed_dim)
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(context_input_dim),
            nn.Linear(context_input_dim, llm_embed_dim * 2),
            nn.GELU(),
            nn.Linear(llm_embed_dim * 2, num_context_tokens * llm_embed_dim),
        )

    def forward(
        self,
        vision_tokens: torch.Tensor,
        intent: torch.Tensor,
        past: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        vision_tokens: (B, N, vision_dim)
        intent: (B,) values in {1, 2, 3}
        past: (B, 16, 6)
        Returns: vision_emb (B, N', llm_embed_dim), context_emb (B, num_context_tokens, llm_embed_dim)
        """
        vision_emb = self._vision_path(vision_tokens)
        context_emb = self._context_path(intent, past)
        return vision_emb, context_emb

    def _vision_path(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample (if configured) then linear proj to llm_embed_dim."""
        n = self.num_vision_tokens_after_downsample
        if n is not None and x.size(1) > n:
            idx = torch.linspace(0, x.size(1) - 1, n, device=x.device, dtype=torch.long)
            x = x[:, idx, :]
        return self.vision_proj(x)

    def _context_path(self, intent: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        """Intent one-hot + past flat -> norm -> MLP -> (B, num_context_tokens, llm_embed_dim)."""
        B = intent.size(0)
        onehot = F.one_hot((intent - 1).long(), num_classes=3).float()
        flat = torch.cat([onehot, past.view(B, -1)], dim=1)
        out = self.context_mlp(flat)
        return out.view(B, self.num_context_tokens, self.llm_embed_dim)
