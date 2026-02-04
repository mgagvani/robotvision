"""
Feature extractors
"""

from typing import List, Union

import torch
import torch.nn as nn
import timm

class FeatureExtractorBase(nn.Module):
    """
    Base interface for feature extractors used in vlm_new.
    Subclasses must set: dims, patch_size, data_config, and implement forward().
    Whether to freeze parameters is decided and applied in the extractor (e.g. frozen=... in __init__).
    """

    dims: List[int]
    patch_size: int
    data_config: dict

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError


class BackboneFeatures(FeatureExtractorBase):
    """
    Minimal timm-based feature extractor. Swap model_name for a different backbone later.
    Freezing is applied here when frozen=True (model does not freeze the extractor).
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        frozen: bool = True,
        features_only: bool = True,
    ):
        super().__init__()
        self._model = timm.create_model(
            model_name, pretrained=True, features_only=features_only
        )
        self.data_config = timm.data.resolve_data_config(model=self._model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        if frozen:
            for p in self._model.parameters():
                p.requires_grad = False

        if features_only:
            self.dims = list(self._model.feature_info.channels())
            # Use last stage reduction as effective patch size
            self.patch_size = self._model.feature_info.reduction()[-1]
        else:
            self.dims = [self._model.num_features]
            self.patch_size = 32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        x_t = self.transforms(x.float())
        feats = self._model(x_t)
        if isinstance(feats, (list, tuple)):
            return list(feats)
        return [feats]
