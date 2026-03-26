import torch
import torch.nn as nn
import torch.nn.functional as F
from .CaRT import ResNetCaRT
from .blocks import TransformerBlock


class PRIX(nn.Module):
    def __init__(
        self,
        out_dim: int = 40,
        *,
        n_cameras: int = 3,
        image_size: tuple[int, int] | None = (256, 384),
        token_grid: tuple[int, int] = (10, 32),
        feature_dim: int = 480,
    ):
        super(PRIX, self).__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even for (x, y) waypoints, got {out_dim}")

        self.out_dim = out_dim
        self.horizon = out_dim // 2
        self.feature_dim = feature_dim
        self.n_cameras = n_cameras
        self.image_size = image_size
        self.token_grid = token_grid
        self.cart = ResNetCaRT()
        self.diffusion_steps = 2
        self.time_embed_dim = feature_dim

        global_ch = self.cart.cart4.out_proj.out_channels
        local_ch = self.cart.fpn.inner_blocks[0].out_channels

        cart_out_ch = global_ch + local_ch
        self.visual_proj = nn.Conv2d(cart_out_ch, self.feature_dim, kernel_size=1, bias=False)

        self.query_init = nn.Linear(3 + 16 * 6, self.feature_dim)

        # ImageNet normalization for res34
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # learnable positional encoding
        n_tokens_per_cam = token_grid[0] * token_grid[1]
        self.n_tokens = n_tokens_per_cam * n_cameras
        self.positional_encoding = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)
        )  # (1, N, C)
        
        # Deep network rather than single attention in MonocularModel 
        self.blocks = nn.ModuleList([
            TransformerBlock(self.feature_dim, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(4)
        ])

        self.anchor_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, out_dim),
        )

        self.path_queries = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((1, self.horizon, self.feature_dim)), std=0.02)
        )
        self.path_pos_emb = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((1, self.horizon, self.feature_dim)), std=0.02)
        )

        self.path_encoder = nn.Sequential(
            nn.Linear(6, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.diffusion_blocks = nn.ModuleList([
            TransformerBlock(self.feature_dim, num_heads=8, mlp_dim=self.feature_dim * 4)
            for _ in range(self.diffusion_steps)
        ])
        self.diffusion_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, 4),
            )
            for _ in range(self.diffusion_steps)
        ])

        betas = torch.tensor([0.08, 0.18], dtype=torch.float32)
        alpha = 1.0 - betas
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bar[:-1]], dim=0)
        self.register_buffer("diffusion_betas", betas, persistent=False)
        self.register_buffer("diffusion_alpha_bar", alpha_bar, persistent=False)
        self.register_buffer("diffusion_alpha_bar_prev", alpha_bar_prev, persistent=False)

    def _preprocess_images(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B, 3, H, W), uint8 from decode_jpeg or float already
        if not torch.is_floating_point(imgs):
            imgs = imgs.float().div(255.0)
        else:
            imgs = imgs.float()

        if self.image_size is not None:
            imgs = F.interpolate(imgs, size=self.image_size, mode="bilinear", align_corners=False)

        imgs = (imgs - self.imagenet_mean) / self.imagenet_std
        return imgs

    def _sample_truncated_noise(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        noise = torch.randn(shape, device=device, dtype=dtype)
        return noise.clamp_(-2.0, 2.0)

    def _time_embedding(self, step_idx: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = self.diffusion_alpha_bar[step_idx].to(device=device, dtype=dtype)
        half_dim = self.time_embed_dim // 2
        freq = torch.exp(
            -torch.arange(half_dim, device=device, dtype=dtype)
            * (torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(half_dim - 1, 1))
        )
        angles = t * freq
        emb = torch.cat([angles.sin(), angles.cos()], dim=0)
        if emb.numel() < self.time_embed_dim:
            emb = F.pad(emb, (0, self.time_embed_dim - emb.numel()))
        return emb.unsqueeze(0).expand(batch_size, -1)

    def _decode_path(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_decoder(query).view(query.size(0), self.horizon, 2)
        noise = self._sample_truncated_noise(anchor.shape, anchor.device, anchor.dtype)
        alpha_bar_t = self.diffusion_alpha_bar[-1].to(device=anchor.device, dtype=anchor.dtype)
        traj = alpha_bar_t.sqrt() * anchor + (1.0 - alpha_bar_t).sqrt() * noise
        self_cond = torch.zeros_like(anchor)

        for step_idx, (block, head) in enumerate(zip(self.diffusion_blocks, self.diffusion_heads)):
            step_embed = self.time_mlp(
                self._time_embedding(step_idx, traj.size(0), traj.device, traj.dtype)
            ).unsqueeze(1)
            path_features = torch.cat([traj, anchor, self_cond], dim=-1)
            path_tokens = self.path_encoder(path_features)
            path_tokens = path_tokens + self.path_queries + self.path_pos_emb
            path_tokens = path_tokens + query.unsqueeze(1)
            path_tokens = path_tokens + step_embed

            refined_tokens = block(path_tokens, tokens)
            noise_pred, delta = head(refined_tokens).chunk(2, dim=-1)

            alpha_bar = self.diffusion_alpha_bar[step_idx].to(device=traj.device, dtype=traj.dtype)
            alpha_bar_prev = self.diffusion_alpha_bar_prev[step_idx].to(device=traj.device, dtype=traj.dtype)
            pred_x0 = (traj - (1.0 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt().clamp_min(1e-6)
            pred_x0 = pred_x0 + delta

            if step_idx < self.diffusion_steps - 1:
                traj = alpha_bar_prev.sqrt() * pred_x0 + (1.0 - alpha_bar_prev).sqrt() * noise_pred
                if self.training:
                    refresh_noise = 0.05 * self._sample_truncated_noise(traj.shape, traj.device, traj.dtype)
                    traj = traj + refresh_noise
            else:
                traj = pred_x0

            self_cond = pred_x0.detach() if self.training else pred_x0

        return traj

    def forward(self, x):
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        if isinstance(images, (list, tuple)):
            camera_list = list(images[: self.n_cameras])
        elif torch.is_tensor(images) and images.ndim == 5:
            # (B, N, 3, H, W) -> list[(B, 3, H, W)]
            n = min(self.n_cameras, images.shape[1])
            camera_list = [images[:, i] for i in range(n)]
        else:
            raise TypeError(f"Unsupported IMAGES type/shape: {type(images)}")

        tokens_per_cam = []
        for cam in camera_list:
            cam = self._preprocess_images(cam)
            feats = self.cart(cam)  # (B, C, H', W')
            feats = F.adaptive_avg_pool2d(feats, output_size=self.token_grid)  # (B, C, Gh, Gw)
            feats = self.visual_proj(feats)  # (B, D, Gh, Gw)
            tokens = feats.flatten(2).permute(0, 2, 1)  # (B, N, D)
            tokens_per_cam.append(tokens)

        tokens = torch.cat(tokens_per_cam, dim=1)  # (B, N_total, D)
        pos = self.positional_encoding[:, : tokens.size(1), :].to(dtype=tokens.dtype)
        tokens = tokens + pos
        
        # copy procedure to build query_0 from MonocularModel
        intent_idx = (intent.long() - 1).clamp(min=0, max=2)
        intent_onehot = F.one_hot(intent_idx, num_classes=3).float()
        past_flat = past.view(past.size(0), -1)
        query = self.query_init(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)

        for block in self.blocks:
            query = block(query, tokens)

        traj = self._decode_path(query.squeeze(1), tokens)
        return traj.reshape(traj.size(0), self.out_dim)
