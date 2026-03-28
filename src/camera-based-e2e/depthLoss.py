# depthLoss.py
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole
import torch
import torch.nn.functional as F
import numpy as np

class DepthLoss:
    def __init__(self, device):
        self.device = device
        self.depth_model = UniDepthV2.from_pretrained(
            "lpiccinelli/unidepth-v2-vitl14"
        ).to(self.device)
        self.depth_model.eval()

    def get_depth(self, images, intrinsics=None):
        """
        Args:
            images: (B, C, H, W) float tensor, pixel values 0-255
            intrinsics: (B, 3, 3) or (3, 3) tensor of camera intrinsics, or None
        Returns:
            (B, H, W) metric depth in meters
        """
        depths = []
        for i in range(images.shape[0]):
            # UniDepth expects (C, H, W), uint8-range floats or uint8
            rgb = images[i].to(self.device)  # (C, H, W)

            if intrinsics is not None:
                K = intrinsics[i] if intrinsics.dim() == 3 else intrinsics
                camera = Pinhole(K=K.unsqueeze(0).to(self.device))
                preds = self.depth_model.infer(rgb, camera)
            else:
                preds = self.depth_model.infer(rgb)

            # preds["depth"] is (1, H, W) metric depth
            depths.append(preds["depth"].squeeze(0))

        return torch.stack(depths, dim=0)  # (B, H, W)

    def compute_depth_loss(self, gt_images, pred_depths, loss_fn):
        pred_depth = self.get_depth(gt_images)
        return loss_fn(pred_depth, pred_depths)

    def __call__(self, gt_images, pred_depths, intrinsics=None, loss_fn=F.l1_loss):
        pred_depth = self.get_depth(gt_images, intrinsics)
        return loss_fn(pred_depth, pred_depths)
