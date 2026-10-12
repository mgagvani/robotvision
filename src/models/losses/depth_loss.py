from transformers import AutoModelForDepthEstimation
import math
import torch
import torch.nn.functional as F

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# DPTImageProcessor config for Depth-Anything-V2-Small:
#   do_resize=True, size=(518, 518), resample=BICUBIC, keep_aspect_ratio=True,
#   ensure_multiple_of=14
#   do_rescale=True  (1/255)
#   do_normalize=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_DPT_TARGET = (518, 518)
_DPT_MULTIPLE = 14


def _dpt_output_size(h: int, w: int) -> tuple[int, int]:
    """Replicates DPTImageProcessor.get_resize_output_image_size exactly."""
    def _constrain(val, multiple, min_val=0, max_val=None):
        x = round(val / multiple) * multiple
        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple
        if x < min_val:
            x = math.ceil(val / multiple) * multiple
        return int(x)

    scale_h = _DPT_TARGET[0] / h
    scale_w = _DPT_TARGET[1] / w
    if abs(1 - scale_w) < abs(1 - scale_h):
        scale_h = scale_w
    else:
        scale_w = scale_h
    return (
        _constrain(scale_h * h, _DPT_MULTIPLE),
        _constrain(scale_w * w, _DPT_MULTIPLE),
    )


class DepthLoss:
    def __init__(self, device):
        self.device = device
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(self.device)
        self.mean = _NORM_MEAN.to(device)
        self.std = _NORM_STD.to(device)
        self._size_cache: dict[tuple[int, int], tuple[int, int]] = {}

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """GPU-native replica of DPTImageProcessor.

        Aspect-ratio-preserving bicubic resize (constrained to multiples of
        14), clamp to [0, 255] (matching PIL uint8 rounding), rescale to
        [0, 1], then ImageNet-normalize.  Antialias is enabled so the resize
        kernel matches PIL/Pillow's BICUBIC filter.
        """
        h, w = images.shape[-2:]
        key = (h, w)
        if key not in self._size_cache:
            self._size_cache[key] = _dpt_output_size(h, w)
        target = self._size_cache[key]

        x = F.interpolate(
            images.float(), size=target,
            mode="bicubic", align_corners=False, antialias=True,
        ).clamp(0, 255)
        return (x / 255.0 - self.mean) / self.std

    def get_depth(self, images):
        """
        Compute ground truth depth from images

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W).           
        """
        pixel_values = self._preprocess(images)

        with torch.no_grad():
            outputs = self.depth_model(pixel_values=pixel_values)
            predicted_depth = outputs.predicted_depth

        height, width = images.shape[2], images.shape[3]

        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        return prediction

    def compute_depth_loss(self, gt_images, pred_depths, loss_fn):
        pred_depth = self.get_depth(gt_images)
        depth_loss = loss_fn(pred_depth, pred_depths)

        return depth_loss
    
    def __call__(self, gt_images, pred_depths, loss_fn=F.l1_loss):
        return self.compute_depth_loss(gt_images, pred_depths, loss_fn)
