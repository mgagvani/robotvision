import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34, ResNet34_Weights
import torch.utils.checkpoint as checkpoint

class OptimizedTransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, use_checkpoint=False):
        super(OptimizedTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        B, L, C = x.shape
        
        x_norm = self.norm1(x)
        
        # Enforce contiguous memory before reshaping
        qkv = self.qkv(x_norm).contiguous().reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = F.scaled_dot_product_attention(q, k, v)
        
        # Enforce contiguous memory after transpose
        attn = attn.transpose(1, 2).contiguous().reshape(B, L, C)
        x = x + self.proj(attn)
        
        x = x + self.mlp(self.norm2(x))
        
        return x

class CaRT(nn.Module):
    def __init__(self, in_channels, shared_transformer, shared_dim=256):
        super(CaRT, self).__init__()
        self.shared_transformer = shared_transformer
        self.shared_dim = shared_dim
        
        # bias=False because the transformer block immediately applies LayerNorm
        self.in_proj = nn.Conv2d(in_channels, shared_dim, kernel_size=1, bias=False)
        
        # Retains bias because it connects to a residual addition, not a norm
        self.out_proj = nn.Conv2d(shared_dim, in_channels, kernel_size=1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_pooled = F.adaptive_avg_pool2d(x, (8, 32))
        x_shared = self.in_proj(x_pooled)
        
        # Enforce contiguous memory before viewing
        x_seq = x_shared.contiguous().view(B, self.shared_dim, -1).permute(0, 2, 1)
        
        x_seq = self.shared_transformer[0](x_seq)
        x_seq_out = self.shared_transformer[1](x_seq)
        
        # Enforce contiguous memory before viewing
        attn_output = x_seq_out.permute(0, 2, 1).contiguous().view(B, self.shared_dim, 8, 32)
        attn_projected = self.out_proj(attn_output)
        
        attn_upsampled = F.interpolate(
            attn_projected, size=(H, W), mode='bilinear', align_corners=False
        )
        
        out = x + attn_upsampled

        out = self.dropout(out)
        
        return out

class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(TopDownFPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            
            # Integrated in-place ReLU for FPN activations
            self.layer_blocks.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))

    def forward(self, features):
        last_inner = self.inner_blocks[-1](features[-1])
        out = self.layer_blocks[-1](last_inner)
        
        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            out = self.layer_blocks[idx](last_inner)
            
        return out

class ResNetCaRT(nn.Module):
    def __init__(
        self,
        fpn_out_channels=256,
        shared_dim=512,
        num_heads=4,
        use_checkpoint=True,
        freeze_resnet: bool = True,
    ):
        super(ResNetCaRT, self).__init__()
        
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_resnet:
            for p in self.stem.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
            for p in self.layer2.parameters():
                p.requires_grad = False
            for p in self.layer3.parameters():
                p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = False
        
        self.shared_transformer = nn.ModuleList([
            OptimizedTransformerBlock(
                embed_dim=shared_dim, 
                num_heads=num_heads,
                use_checkpoint=use_checkpoint
            ),
            OptimizedTransformerBlock(
                embed_dim=shared_dim, 
                num_heads=num_heads,
                use_checkpoint=use_checkpoint
            )
        ])
        
        self.cart1 = CaRT(64, self.shared_transformer, shared_dim)
        self.cart2 = CaRT(128, self.shared_transformer, shared_dim)
        self.cart3 = CaRT(256, self.shared_transformer, shared_dim)
        self.cart4 = CaRT(512, self.shared_transformer, shared_dim)
        
        self.fpn = TopDownFPN(in_channels_list=[64, 128, 256, 512], out_channels=fpn_out_channels)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
        x0 = self.stem(x)
        
        x1 = self.layer1(x0)
        x1_c = self.cart1(x1)
        
        x2 = self.layer2(x1_c)
        x2_c = self.cart2(x2)
        
        x3 = self.layer3(x2_c)
        x3_c = self.cart3(x3)
        
        x4 = self.layer4(x3)
        x4_c = self.cart4(x4)
        
        global_features = x4_c
        local_features = self.fpn([x1_c, x2_c, x3_c, x4_c])
        
        global_features_upsampled = F.interpolate(
            global_features, 
            size=local_features.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        visual_features = torch.cat([global_features_upsampled, local_features], dim=1)

        visual_features = self.dropout(visual_features)
        
        return visual_features
