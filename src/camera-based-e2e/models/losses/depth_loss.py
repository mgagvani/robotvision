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
    

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Must run on GPU")

    import sys
    from pathlib import Path
    # Determine the absolute path to the directory containing loader.py
    # depth_loss.py is in models/losses/
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    sys.path.append(str(project_root))

    from loader import WaymoE2E
    loader = WaymoE2E(indexFile="index_val.pkl", data_dir="/anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0/")
    data_iterator = iter(torch.utils.data.DataLoader(loader, batch_size=8, num_workers=4))

    device = torch.device("cuda")
    depth_loss_fn = DepthLoss(device)

    from matplotlib import pyplot as plt
    import numpy as np

    for _ in range(6):
        batch = next(data_iterator)
    images = batch["IMAGES"][1].to(device)  # front camera

    res = depth_loss_fn.get_depth(images)  # (B, H, W) tensor

    fig, ax = plt.subplots(8, 2, figsize=(16, 48))
    for i in range(8):
        # Input image
        ax[i, 0].set_title(f"Input Image {i+1}")
        ax[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        ax[i, 0].axis('off')
        
        # Predicted depth
        ax[i, 1].set_title(f"Predicted Depth {i+1}")
        depth_img = ax[i, 1].imshow(res[i].cpu().numpy(), cmap='plasma')
        ax[i, 1].axis('off')
        fig.colorbar(depth_img, ax=ax[i, 1], orientation='vertical', label='Depth')
    
    plt.tight_layout()
    fig.savefig("predicted_depth.png", dpi=150, bbox_inches='tight')

    # throughput test
    from time import perf_counter

    start_time = perf_counter()
    times = []
    for _ in range(100):
        batch = next(data_iterator)
        images = batch["IMAGES"][1].to(device)  # front camera
        t0 = perf_counter()
        res = depth_loss_fn.get_depth(images)
        times.append(perf_counter() - t0)
    end_time = perf_counter()
    print(f"Throughput: {100 / (end_time - start_time):.2f} batches/sec")
    print(f"Avg depth inference batch/s: {1 / np.mean(times):.2f} FPS")

