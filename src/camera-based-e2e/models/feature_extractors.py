from typing import List
import torch
import torch.nn as nn
import timm
import torchvision
from torchvision.transforms import v2

class DINOFeatures(nn.Module):
    def __init__(self, model_name: str = "vit_small_plus_patch16_dinov3.lvd1689m", frozen: bool = True):
        super(DINOFeatures, self).__init__()

        self.dino_model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.data_config = timm.data.resolve_data_config(model=self.dino_model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        if frozen:
            for param in self.dino_model.parameters():
                param.requires_grad = False

        self.dims = [384, 384, 384]  # feature dims for each layer
        self.patch_size = 16  # patch size

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        # transforms: resize 256x256, center crop, normalize
        x_t = self.transforms(x.float().div(255.0)) # preprocess
        features = self.dino_model(x_t)
        return features # 3 x [B, 384, 16, 16]
    
class SAMFeatures(nn.Module):
    def __init__(
        self,
        model_name: str = "timm/sam2_hiera_tiny.fb_r896_2pt1",
        frozen: bool = True,
        feature_stage: int = -1,
    ):
        super(SAMFeatures, self).__init__()

        # features_only returns a list of stage outputs
        self.sam_model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.data_config = timm.data.resolve_data_config(model=self.sam_model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        if frozen:
            for param in self.sam_model.parameters():
                param.requires_grad = False
            self.sam_model.eval()

        channels = self.sam_model.feature_info.channels()
        reductions = self.sam_model.feature_info.reduction()
        self.feature_stage = feature_stage
        self.dims = [channels[feature_stage]]
        self.patch_size = reductions[feature_stage]  # effective stride

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        x_t = self.transforms(x.float().div(255.0))  # preprocess
        feats = self.sam_model(x_t)       # list of feature maps
        return [feats[self.feature_stage]] # (B, C, H', W')

class EUPEFeatures(nn.Module):
    EUPE_DIR = "/depot/mlp/data/robotvision/eupe" # Works on Gilbreth, Gautschi, Negishi, change if on Anvil
    CHECKPOINT_PATH = f"{EUPE_DIR}/EUPE-ViT-S.pt"
    SIZE = (768, 768) # pe_spatial_... is 512x512 I believe

    def make_transform(self, resize_size: int = 256):
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])

    def __init__(self, model_name: str = "EUPE-ViT-S", frozen: bool = True, feature_stage: int = -1):
        super(EUPEFeatures, self).__init__()
        if feature_stage != -1:
            raise NotImplementedError
        self.transform = self.make_transform(resize_size=self.SIZE[0]) # pics are like 900 x 1000
        self.model = torch.hub.load(self.EUPE_DIR, "eupe_vits16", source="local", weights=self.CHECKPOINT_PATH)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self.dims = [384]  # feature dim for the last layer
        self.patch_size = 16
        self.n_tokens = (self.SIZE[0] // self.patch_size) ** 2 # 1024
        self.data_config = {
            "input_size": (3, self.SIZE[0], self.SIZE[1])
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        x_t = self.transform(x)  # preprocess
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                features = self.model.forward_features(x_t) # list of feature maps
        clstoken, patchtokens = features["x_norm_clstoken"], features["x_norm_patchtokens"]
        B, N, C = patchtokens.shape
        H = W = int(N**0.5)
        patchtokens = patchtokens.transpose(1, 2).reshape(B, C, H, W) # (B, C, H', W')
        return [patchtokens]