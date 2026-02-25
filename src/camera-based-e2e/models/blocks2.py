from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = sqrt(self.head_dim)
        self.use_rope = use_rope

        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError("RoPE requires an even head dimension.")
        if self.use_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().view(1, 1, seq_len, self.head_dim).to(dtype=x.dtype)
        sin = emb.sin().view(1, 1, seq_len, self.head_dim).to(dtype=x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, q_tokens, channels = x.shape
        context = context if context is not None else x
        k_tokens = context.shape[1]

        q = self.q_proj(x).view(bsz, q_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(bsz, k_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(bsz, k_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self._apply_rope(q, q_tokens)
            k = self._apply_rope(k, k_tokens)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(bsz, q_tokens, channels)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, use_rope_self_attn=False, use_rope_cross_attn=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = MHA(embed_dim, num_heads, use_rope=use_rope_self_attn)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MHA(embed_dim, num_heads, use_rope=use_rope_cross_attn)

        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        query = query + self.self_attn(self.ln1(query))
        query = query + self.cross_attn(self.ln2(query), context=tokens)
        query = query + self.mlp(self.ln3(query))
        return query
