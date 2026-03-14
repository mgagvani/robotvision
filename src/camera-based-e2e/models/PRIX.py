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
        self.feature_dim = feature_dim
        self.n_cameras = n_cameras
        self.image_size = image_size
        self.token_grid = token_grid
        self.cart = ResNetCaRT()

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
        
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, out_dim),
        )

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

        return self.decoder(query.squeeze(1))
