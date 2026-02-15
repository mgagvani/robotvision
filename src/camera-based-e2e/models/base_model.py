import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

from .losses.depth_loss import DepthLoss

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
    def __init__(self, model: nn.Module, lr: float, lr_vision: float | None = None, rfs_weight: float = 0.0):
        super(LitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr
        self.hparams.lr_vision = lr_vision
        self.hparams.rfs_weight = rfs_weight

        self.example_input_array = ({
            'PAST': torch.zeros((1, 16, 6)),  # PAST
            'IMAGES': [torch.zeros((1, 3, 1280, 1920)) for _ in range(6)],  # IMAGES
            'INTENT': torch.tensor([1.0]),  # INTENT
        },)

        self.save_hyperparameters()

    # --- Data Loading ---- 
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # move actual tensors
        out = super().transfer_batch_to_device(batch, device, dataloader_idx)
        # keep encoded JPEG bytes on cpu to decode later
        if "IMAGES_JPEG" in batch:
            out["IMAGES_JPEG"] = batch["IMAGES_JPEG"]
        return out


    def decode_batch_jpeg(self, images_jpeg: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        # Flatten cameras
        flat_encoded, cam_sizes = [], []
        for cam in images_jpeg:
            cam_sizes.append(len(cam))
            flat_encoded.extend(
                jpg if isinstance(jpg, torch.Tensor) else torch.frombuffer(memoryview(jpg), dtype=torch.uint8)
                for jpg in cam
            )
        
        flat_decoded = torchvision.io.decode_jpeg(
            flat_encoded, 
            mode=torchvision.io.ImageReadMode.UNCHANGED,
            device = self.device,
        ) # list of (C, H, W) gpu tensors

        out = []
        idx = 0
        for n in cam_sizes:
            cam_list = flat_decoded[idx: idx+n]
            idx += n
            out.append(torch.stack(cam_list, dim=0))  # (B, C, H, W)
        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.depth_loss = DepthLoss(self.device)

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory
        """
        return torch.mean(torch.norm(pred - gt, dim=-1))

    def time_thresholds(self, t_idx):
        # Time-based thresholds at 3s and 5s.
        lat = torch.where(t_idx <= 3, 1.0, 1.8)
        lng = torch.where(t_idx <= 3, 4.0, 7.2)
        return lat, lng

    def speed_scale(self, v):
        # Speed-based scaling copied from RFS paper.
        return torch.where(
            v < 1.4,
            0.5,
            torch.where(
                v < 11.0,
                0.5 + 0.5 * (v - 1.4) / (11.0 - 1.4),
                1.0
            )
        )

    def compute_direction(self, trajectory):
        # Pad with first point so displacement stays (B, T, 2).
        padded = torch.cat([trajectory[:, :1], trajectory], dim=1)
        displacement = padded[:, 1:] - padded[:, :-1]
        lng_dir = F.normalize(displacement, p=2, dim=-1, eps=1e-6)
        lat_dir = torch.stack([-lng_dir[..., 1], lng_dir[..., 0]], dim=-1)
        return lng_dir, lat_dir

    def rfs_loss(self, pred, gt, lng_dir, lat_dir, speed, t_idx):
        """
        pred, gt: (B, T, 2)
        speed: (B,) or (B, T)
        t_idx: (T,) or (B, T)
        """
        delta = pred - gt
        delta_lng = (delta * lng_dir).sum(dim=-1).abs()
        delta_lat = (delta * lat_dir).sum(dim=-1).abs()

        tau_lat_raw, tau_lng_raw = self.time_thresholds(t_idx)
        scale = self.speed_scale(speed)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)

        tau_lat = tau_lat_raw * scale
        tau_lng = tau_lng_raw * scale

        deviation = torch.max(
            delta_lat / tau_lat,
            delta_lng / tau_lng,
        )
        score = torch.where(
            deviation <= 1,
            torch.ones_like(deviation),
            torch.pow(0.1, deviation - 1)
        )
        return (1.0 - score).mean()

    def _prepare_rfs_inputs(self, past, future, pred_future):
        speed = torch.norm(past[..., 2:4], dim=-1)[:, -1]  # (B,), speed at last observed time step
        full_lng_dir, full_lat_dir = self.compute_direction(future)
        indices = [11, 19]  # 3s and 5s into the future

        pred_slice = pred_future[:, indices, :]
        gt_slice = future[:, indices, :]
        lng_dir_slice = full_lng_dir[:, indices, :]
        lat_dir_slice = full_lat_dir[:, indices, :]
        t_idx = torch.tensor([3.0, 5.0], device=future.device).unsqueeze(0).expand(future.size(0), -1)
        return pred_slice, gt_slice, lng_dir_slice, lat_dir_slice, speed, t_idx
    
    # ---- optimizers ----
    def configure_optimizers(self):
        # NOTE: This can be extended and tuned, LR especially will differ and have an impact.
        # vision encoder, if trainable, should have 1/10 the LR of the rest of the model
        if hasattr(self.model, "features"):
            encoder_params = [p for p in self.model.features.parameters() if p.requires_grad]
            other_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith("features.") and p.requires_grad
            ]
            if encoder_params:
                encoder_lr = self.hparams.lr * 0.1 if self.hparams.lr_vision is None else self.hparams.lr_vision
                return torch.optim.Adam(
                    [
                        {"params": other_params, "lr": self.hparams.lr},
                        {"params": encoder_params, "lr": encoder_lr},
                    ]
                )
            if other_params:
                return torch.optim.Adam(other_params, lr=self.hparams.lr)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    # ---- forward / step ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, intent = batch['PAST'], batch['FUTURE'], batch['INTENT']

        if "IMAGES_JPEG" in batch:
            images_jpeg = batch["IMAGES_JPEG"]
            images = self.decode_batch_jpeg(images_jpeg)
        elif "IMAGES" in batch:
            images = batch["IMAGES"]
        else:
            raise KeyError("Batch must contain either 'IMAGES_JPEG' or 'IMAGES' key.")
        
        # `past` is our input (B, 16, 6) e.g. Batch x Time x (x, y, v_x, v_y, a_x, a_y)
        # and `future` is our output (B, 20, 2) e.g. Batch x Time x (x, y)

        # create all input data that we are allowed to give to a model
        model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}

        pred_future = self.forward(model_inputs)  # (B, T*2)
        pred_depth = None
        pred_scores: torch.Tensor = None
        if isinstance(pred_future, dict):
            pred_future, pred_depth, pred_scores = pred_future["trajectory"], pred_future.get("depth", None), pred_future.get("scores", None)

        pred = pred_future
        t_steps = future.shape[1]
        t2 = t_steps * 2
        k_modes = self.model.n_proposals if hasattr(self.model, "n_proposals") else 1

        if pred.ndim != 2:
            raise ValueError(f"Unexpected pred shape {pred.shape}; expected (B, T*2) or (B, {k_modes}*T*2).")

        if pred.shape[1] == t2:
            pred = pred.view(pred.size(0), 1, t_steps, 2)
        elif pred.shape[1] == k_modes * t2:
            pred = pred.view(pred.size(0), k_modes, t_steps, 2)
        else:
            raise ValueError(f"Unexpected pred shape {pred.shape}; expected (B, T*2) or (B, {k_modes}*T*2).")

        if pred_scores is not None and pred.size(1) > 1:
            rfs_pred_idx = pred_scores.argmin(dim=1)
        else:
            rfs_pred_idx = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)
        pred_for_rfs = pred[torch.arange(pred.size(0), device=pred.device), rfs_pred_idx]

        pred_slice, gt_slice, lng_dir_slice, lat_dir_slice, speed, t_idx = self._prepare_rfs_inputs(
            past,
            future,
            pred_for_rfs,
        )
        rfs_unweighted = self.rfs_loss(pred_slice, gt_slice, lng_dir_slice, lat_dir_slice, speed, t_idx)
        rfs_weight = getattr(self.hparams, "rfs_weight", 0.0)
        loss_rfs = rfs_weight * rfs_unweighted

        # ADE per mode: (B, K)
        dist = torch.norm(pred - future[:, torch.newaxis, :, :], dim=-1)  # (B, K, T)
        ade_per_mode = dist.mean(dim=-1)

        # Top-M WTA for trajectory loss. Here, we have an "oracle" that picks the best mode
        # so, our loss is calculated on the mean of the top 5 trajectories.
        top_m = min(5, ade_per_mode.size(1))
        loss_ade = ade_per_mode.topk(top_m, largest=False, dim=1).values.mean()

        # oracle ade is best of all proposals, since we have the GT data during training
        oracle_ade = ade_per_mode.min(dim=1).values.mean()
        ade_pred = None
        # pred_scores is now the predicted ADE of each trajectory
        if pred_scores is not None and k_modes > 1:
            pred_idx = pred_scores.argmin(dim=1)
            ade_pred = ade_per_mode[torch.arange(pred.size(0), device=pred.device), pred_idx].mean()
        elif k_modes == 1:
            ade_pred = ade_per_mode.squeeze(1).mean()
        regret = (ade_pred - oracle_ade) if ade_pred is not None else None

        # Scorer Losses -> encourage ranking of predicted scores to match true ranking of ades that are generated
        if k_modes > 1 and pred_scores is not None:
            ade = ade_per_mode.detach()  # (B, K)
            loss_score = F.mse_loss(pred_scores, ade)
            pred_idx = pred_scores.argmin(dim=1)
            ade_pred = ade_per_mode[torch.arange(pred.size(0), device=pred.device), pred_idx].mean()
        else:
            loss_score = torch.tensor(0.0, device=self.device)

        # Depth Loss
        if pred_depth is not None:
            front_img = images[1]  # front camera
            depth_in = F.interpolate(front_img, size=(128, 128), mode='nearest')
            loss_depth = self.depth_loss(depth_in, pred_depth, loss_fn=F.l1_loss)
        else:
            loss_depth = torch.tensor(0.0, device=self.device)

        loss_depth *= 0.1 # slightly enabled
        loss_ade *= 1.0 # TODO: tune loss terms
        loss_score *= 1.0
        total_loss = loss_ade + loss_depth + loss_score + loss_rfs
        # TODO: improve logging both to disk and to console
        log_payload = {
            f"{stage}_loss_ade": loss_ade,
            f"{stage}_loss_score": loss_score,
            f"{stage}_loss_depth": loss_depth,
            f"{stage}_loss_rfs": loss_rfs,
            f"{stage}_rfs_unweighted": rfs_unweighted,
            f"{stage}_loss": total_loss,
        }
        if ade_pred is not None:
            log_payload[f"{stage}_ade_pred"] = ade_pred
            log_payload[f"{stage}_ade_oracle"] = oracle_ade
            log_payload[f"{stage}_ade_regret"] = regret
        self.log_dict(log_payload, prog_bar=True, logger=True)

        return total_loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")
    
def collate_with_images(batch):
    past = [torch.as_tensor(b["PAST"], dtype=torch.float32) for b in batch]
    future = [torch.as_tensor(b["FUTURE"], dtype=torch.float32) for b in batch]
    intent = torch.as_tensor([b["INTENT"] for b in batch])
    names = [b["NAME"] for b in batch]

    cams = list(zip(*[b["IMAGES_JPEG"] for b in batch]))  # per-camera tuples
    images_jpeg = [list(cam_imgs) for cam_imgs in cams]  # stay on CPU

    return {
        "PAST": torch.stack(past, dim=0),
        "FUTURE": torch.stack(future, dim=0),
        "INTENT": intent,
        "IMAGES_JPEG": images_jpeg,
        "NAME": names,
    }
