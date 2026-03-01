from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        context = context if context is not None else x
        M = context.shape[1] # Number of context tokens

        # Project and split into heads: (B, N, C) -> (B, N, H, D) -> (B, H, N, D)
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, H, N, D) @ (B, H, D, M) -> (B, H, N, M)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        # Combine (B, H, N, M) @ (B, H, M, D) -> (B, H, N, D)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    '''
    Combined self (query <-> query) and cross (query <-> tokens) attention block 
    '''
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = MHA(embed_dim, num_heads)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MHA(embed_dim, num_heads)
        
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # self attention
        query = query + self.self_attn(self.ln1(query))
        
        # cross attention (Query <-> Tokens)
        # Note: tokens do not get updated here
        query = query + self.cross_attn(self.ln2(query), context=tokens)
        
        # 3. feed forward to next block (MLP)
        query = query + self.mlp(self.ln3(query))
        return query
    
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_norm_q = nn.LayerNorm(dim)
        self.cross_attn_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden_features = int(dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self‑attention
        residual = x
        x_norm = self.self_attn_norm(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + attn_output

        # Cross‑attention
        residual = x
        q = self.cross_attn_norm_q(x)
        k = self.cross_attn_norm_kv(context)
        v = k
        attn_output, _ = self.cross_attn(q, k, v)
        x = residual + attn_output

        # Feed‑forward
        residual = x
        x_norm = self.mlp_norm(x)
        x = residual + self.mlp(x_norm)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> List[torch.Tensor]:
        hidden_states: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, context)
            hidden_states.append(x)
        return hidden_states


class TransformerDecoderScorer(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context)
        return x

class CrossAttention(nn.Module):
    '''
    Unlike the nn.TransformerDecoderLayer, there is no self attention (Q' = Attention(Q, Q, Q)) here so
    we do not incur the quadratic complexity of each of the proposals attending to each other.
    '''
    def __init__(self, d_model: int, n_head: int, d_ffn: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.0)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # q: (B, K, d), mem: (B, S, d)
        q_norm = self.ln_q(q)
        attn_out, _ = self.cross(q_norm, mem, mem, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.ln_ff(q))
        return q