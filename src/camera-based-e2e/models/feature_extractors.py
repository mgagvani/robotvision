from typing import List
import torch
import torch.nn as nn
import timm

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
        return [feats[self.feature_stage]]