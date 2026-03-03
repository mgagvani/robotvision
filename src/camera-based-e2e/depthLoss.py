from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import torch.nn.functional as F

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

class DepthLoss:
    def __init__(self, device):
        self.device = device
        self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(self.device)

    def get_depth(self, images):
        """
        Compute ground truth depth from images

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W).           
        """
        # Preprocess images
        inputs = self.depth_processor(images=images, return_tensors="pt").to(self.device)

        # Forward pass through the depth estimation model
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        height, width = images.shape[2], images.shape[3]
        
        # Interpolate to original size
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
    loader = WaymoE2E(indexFile="index_val.pkl", data_dir="/anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0/", images=True)
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