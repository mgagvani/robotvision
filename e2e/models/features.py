import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VitFeatures(nn.Module):
    def __init__(
            self, 
            model_name='vit_tiny_patch16_224.augreg_in21k',
            frozen=True
        ):
        super(VitFeatures, self).__init__()

        # Use pretrained weights by default; freeze controls finetuning, not whether to load weights.
        self.vit = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
        )

        self.data_config = timm.data.resolve_data_config(model=self.vit)

        self.mean = torch.tensor(self.data_config['mean']).view(1, 3, 1, 1)
        self.std = torch.tensor(self.data_config['std']).view(1, 3, 1, 1)
        self.input_size = self.data_config['input_size'][1:]

        # Feature dims and effective patch size for each stage.
        channels = self.vit.feature_info.channels()
        reductions = self.vit.feature_info.reduction()


        # Use the last stage by default (highest-level features).
        self.feature_stage = -1
        self.dims = [channels[self.feature_stage]]
        self.patch_size = reductions[self.feature_stage]

        if frozen:
            for param in self.vit.parameters():
                param.requires_grad = False        

    def forward(self, x: torch.Tensor):

        x = x.float() / 255.0
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        feats = self.vit(x)

        return [feats[self.feature_stage]]