import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pytorch_lightning as pl

class BaseModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BaseModel, self).__init__()
        
        # This is literally just linear regression = y_hat = Wx + b
        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        
    def forward(self, x: dict) -> torch.Tensor:
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        x = past.reshape(past.size(0), -1)  # Flatten to (B, 16 * 6) = (B, 96)
        return self.nn(x)
    
class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super(LitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr

        self.example_input_array = ({
            'PAST': torch.zeros((1, 16, 6)),  # PAST
            'IMAGES': [torch.zeros((1, 3, 1280, 1920)) for _ in range(6)],  # IMAGES
            'INTENT': torch.tensor([1.0]),  # INTENT
            'FUTURE': torch.zeros((1, 20, 2))
        },)

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory
        """
        return torch.mean(torch.norm(pred - gt, dim=-1))
    
    # ---- optimizers ----
    def configure_optimizers(self):
        # NOTE: This can be extended and tuned, LR especially will differ and have an impact.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    # ---- forward / step ----
    def forward(self, x: torch.Tensor, stage="val") -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']
        
        # `past` is our input (B, 16, 6) e.g. Batch x Time x (x, y, v_x, v_y, a_x, a_y)
        # and `future` is our output (B, 20, 2) e.g. Batch x Time x (x, y)

        # create all input data that we are allowed to give to a model
        model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent, 'FUTURE': future}

        pred_future = self.forward(model_inputs)  # (B, T*2)
        loss = self.ade_loss(pred_future.reshape_as(future), future)  # reshape to (B, T, 2

        # TODO: improve logging both to disk and to console
        self.log_dict({
            f"{stage}_loss": loss,
        }, prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

class TrajectoryNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        
        # 1. Define a linear schedule for beta
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device="cuda")
        
        # 2. Calculate alphas and alpha-bars
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, noise, t):
        """
        Jump from clean trajectory x_0 to noisy trajectory x_t
        """
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None]
        
        # The core formula: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t


class DiffuseLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super(DiffuseLitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr
        self.noise_sched = TrajectoryNoiseScheduler()

        self.example_input_array = ({
            'PAST': torch.zeros((1, 16, 6)),  # PAST
            'IMAGES': [torch.zeros((1, 3, 1280, 1920)) for _ in range(6)],  # IMAGES
            'INTENT': torch.tensor([1.0]),  # INTENT
            'FUTURE': torch.zeros((1, 20, 2))
        },)

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory
        """
        return torch.mean(torch.norm(pred - gt, dim=-1))
    
    # ---- optimizers ----
    def configure_optimizers(self):
        # NOTE: This can be extended and tuned, LR especially will differ and have an impact.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    # ---- forward / step ----
    def forward(self, x: torch.Tensor, t=torch.tensor(0 ), stage="val") -> torch.Tensor:
        return self.model(x, t, stage)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']
        
        # `past` is our input (B, 16, 6) e.g. Batch x Time x (x, y, v_x, v_y, a_x, a_y)
        # and `future` is our output (B, 20, 2) e.g. Batch x Time x (x, y)

        t = torch.randint(0, 1000, (past.size(0),), device=past.device).long()
        noise = torch.randn_like(future, device=future.device)

        if stage == "train":
            future_noisy = self.noise_sched.add_noise(future, noise, t)
            # create all input data that we are allowed to give to a model
            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent, 'FUTURE': future_noisy}
            pred_noise = self.forward(model_inputs, t, stage)  # (B, T*2)
            loss = F.mse_loss(pred_noise.reshape_as(noise), noise)  # reshape to (B, T, 2
        else:
            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent, 'FUTURE': future}
            pred = self.forward(model_inputs, 0, stage)
            loss = self.ade_loss(pred.reshape_as(future), future)

        # TODO: improve logging both to disk and to console
        self.log_dict({
            f"{stage}_loss": loss,
        }, prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")


def collate_with_images(batch):
    past = [torch.as_tensor(b["PAST"], dtype=torch.float32) for b in batch]
    future = [torch.as_tensor(b["FUTURE"], dtype=torch.float32) for b in batch]
    intent = torch.as_tensor([b["INTENT"] for b in batch])
    names = [b["NAME"] for b in batch]

    cams = list(zip(*[b["IMAGES"] for b in batch]))  # per-camera tuples
    images = [torch.stack(cam_imgs, dim=0) for cam_imgs in cams]  # stay on CPU

    return {
        "PAST": torch.stack(past, dim=0),
        "FUTURE": torch.stack(future, dim=0),
        "INTENT": intent,
        "IMAGES": images,
        "NAME": names,
    }


def collate_with_images_tokens_depth(batch):
    past = [torch.as_tensor(b["PAST"], dtype=torch.float32) for b in batch]
    future = [torch.as_tensor(b["FUTURE"], dtype=torch.float32) for b in batch]
    intent = torch.as_tensor([b["INTENT"] for b in batch])
    names = [b["NAME"] for b in batch]

    cams = list(zip(*[b["IMAGES"] for b in batch])) if batch[0]["IMAGES"] else []
    images = [torch.stack(cam_imgs, dim=0) for cam_imgs in cams]

    out = {
        "PAST": torch.stack(past, dim=0),
        "FUTURE": torch.stack(future, dim=0),
        "INTENT": intent,
        "IMAGES": images,
        "NAME": names,
    }
    
    if "TOKENS" in batch[0] and batch[0].get("TOKENS"):
        tokens_batch = [b["TOKENS"] for b in batch] 
        
        stacked_tokens = []
        for cam_idx in range(3):
            cam_tensors = []
            for tokens_list in tokens_batch:
                if tokens_list and cam_idx < len(tokens_list):
                    cam_tensors.append(tokens_list[cam_idx])
                else:
                    cam_tensors.append(torch.zeros(1024, 420))
                    print("FAILED TO LOAD TOKENS FOR CAMERA")
            stacked_tokens.append(torch.stack(cam_tensors, dim=0))
        out["TOKENS"] = stacked_tokens

    # Handle precomputed depth: [tensor_cam0, tensor_cam1, tensor_cam2]
    if "PRECOMPUTED_DEPTH" in batch[0]:
        depth_batch = [b["PRECOMPUTED_DEPTH"] for b in batch]  # List of lists
        
        # Stack along camera dimension: (batch, 3, H, W)
        stacked = []
        for cam_idx in range(3):
            cam_tensors = []
            for depth_list in depth_batch:
                if depth_list and cam_idx < len(depth_list):
                    cam_tensors.append(depth_list[cam_idx])
                else:
                    cam_tensors.append(torch.zeros(128, 128))
                    print("FAILED TO LOAD DEPTH FOR CAMERA")
            stacked.append(torch.stack(cam_tensors, dim=0))
        out["PRECOMPUTED_DEPTH"] = torch.stack(stacked, dim=1)
    
    # Add object detection ground truth if available
    # Stack along camera dimension: (batch, 3, max_detections, 4) for each detection component
    if "PRECOMPUTED_OBJ_DET" in batch[0] and batch[0]["PRECOMPUTED_OBJ_DET"] is not None:
        obj_det_batch = [b["PRECOMPUTED_OBJ_DET"] for b in batch]
        
        # Find max detections across all samples and cameras
        max_det = 0
        for obj_det_list in obj_det_batch:
            if obj_det_list:
                for det in obj_det_list:
                    if det is not None and 'boxes' in det:
                        # Squeeze batch dim if present (shape is 1, num_det, 4)
                        boxes = det['boxes']
                        if boxes.ndim == 3 and boxes.shape[0] == 1:
                            boxes = boxes.squeeze(0)
                        n = boxes.shape[0] if boxes.numel() > 0 else 0
                        max_det = max(max_det, n)
        
        # Default to at least 1 detection slot
        max_det = max(max_det, 1)
        
        # Pad and stack boxes: (batch, 3, max_detections, 4)
        stacked_boxes = []
        stacked_labels = []
        stacked_scores = []
        
        for cam_idx in range(3):
            cam_boxes = []
            cam_labels = []
            cam_scores = []
            for obj_det_list in obj_det_batch:
                if obj_det_list and cam_idx < len(obj_det_list):
                    det = obj_det_list[cam_idx]
                    if det is not None:
                        boxes = det['boxes'] if 'boxes' in det else torch.zeros(0, 4)
                        labels = det['labels'] if 'labels' in det else torch.zeros(0, dtype=torch.long)
                        scores = det['scores'] if 'scores' in det else torch.zeros(0)
                        
                        # Squeeze batch dim if present (shape is 1, num_det, 4)
                        if boxes.ndim == 3 and boxes.shape[0] == 1:
                            boxes = boxes.squeeze(0)
                        if labels.ndim == 2 and labels.shape[0] == 1:
                            labels = labels.squeeze(0)
                        if scores.ndim == 2 and scores.shape[0] == 1:
                            scores = scores.squeeze(0)
                        
                        n = boxes.shape[0] if boxes.numel() > 0 else 0
                        if n < max_det:
                            # Pad to max_det
                            boxes_padded = torch.zeros(max_det, 4)
                            labels_padded = torch.full((max_det,), -1, dtype=torch.long)
                            scores_padded = torch.zeros(max_det)
                            if n > 0:
                                boxes_padded[:n] = boxes
                                labels_padded[:n] = labels
                                scores_padded[:n] = scores
                            cam_boxes.append(boxes_padded)
                            cam_labels.append(labels_padded)
                            cam_scores.append(scores_padded)
                        else:
                            cam_boxes.append(boxes[:max_det])
                            cam_labels.append(labels[:max_det])
                            cam_scores.append(scores[:max_det])
                    else:
                        cam_boxes.append(torch.zeros(max_det, 4))
                        cam_labels.append(torch.full((max_det,), -1, dtype=torch.long))
                        cam_scores.append(torch.zeros(max_det))
                else:
                    cam_boxes.append(torch.zeros(max_det, 4))
                    cam_labels.append(torch.full((max_det,), -1, dtype=torch.long))
                    cam_scores.append(torch.zeros(max_det))
            stacked_boxes.append(torch.stack(cam_boxes, dim=0))
            stacked_labels.append(torch.stack(cam_labels, dim=0))
            stacked_scores.append(torch.stack(cam_scores, dim=0))
        
        out["PRECOMPUTED_OBJ_DET"] = {
            'boxes': torch.stack(stacked_boxes, dim=1),  # (batch, 3, max_det, 4)
            'labels': torch.stack(stacked_labels, dim=1),  # (batch, 3, max_det)
            'scores': torch.stack(stacked_scores, dim=1),  # (batch, 3, max_det)
        }
    
    # Add lane detection ground truth if available
    # Stack along camera dimension: (batch, 3, 2, H, W)
    if "PRECOMPUTED_LANE" in batch[0] and batch[0]["PRECOMPUTED_LANE"] is not None:
        lane_batch = [b["PRECOMPUTED_LANE"] for b in batch]
        
        stacked_lanes = []
        for cam_idx in range(3):
            cam_tensors = []
            for lane_list in lane_batch:
                if lane_list and cam_idx < len(lane_list):
                    cam_tensors.append(lane_list[cam_idx])
                else:
                    cam_tensors.append(torch.zeros(2, 128, 128))
            stacked_lanes.append(torch.stack(cam_tensors, dim=0))
        
        out["PRECOMPUTED_LANE"] = torch.stack(stacked_lanes, dim=1)  # (batch, 3, 2, 128, 128)

    return out
