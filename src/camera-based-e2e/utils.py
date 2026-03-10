import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        original_shape = time.shape
        time = time.flatten()
        
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        if len(original_shape) > 1:
            embeddings = embeddings.view(*original_shape, -1)
        
        return embeddings

def get_sinusoidal_embeddings(n_waypoints, d_model):
    # n_waypoints = 16, d_model = token dimension (e.g., 128)
    pe = torch.zeros(n_waypoints, d_model)
    position = torch.arange(0, n_waypoints, dtype=torch.float).unsqueeze(1)
    
    # The denominator term (10000^(2i/d_model))
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # Fill sine for even indices, cosine for odd
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

def get_2d_sinusoidal_embeddings(height, width, d_model):
    if d_model % 4 != 0:
        raise ValueError("2D sinusoidal embeddings require d_model divisible by 4.")
    half_dim = d_model // 2
    y_embed = get_sinusoidal_embeddings(height, half_dim)  # (H, D/2)
    x_embed = get_sinusoidal_embeddings(width, half_dim)   # (W, D/2)
    y_grid = y_embed[:, None, :].expand(height, width, half_dim)
    x_grid = x_embed[None, :, :].expand(height, width, half_dim)
    return torch.cat([y_grid, x_grid], dim=-1).reshape(height * width, d_model)
