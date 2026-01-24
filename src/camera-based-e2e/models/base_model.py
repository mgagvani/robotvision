import torch
import torch.nn as nn
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
        },)

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory
        """
        return torch.mean(torch.norm(pred - gt, dim=-1))
    
    def time_thresholds(self, t_idx):
        # loss function defines time-based thresholds at 3s and 5s

        lat = torch.where(t_idx <= 3, 1.0, 1.8)
        lng = torch.where(t_idx <= 3, 4.0, 7.2)
        return lat, lng
    
    def speed_scale(self, v):
        # speed-based scaling function, copied from paper
        return torch.where(
            v < 1.4, 
            0.5,
            torch.where(
                v < 11.0,
                0.5 + 0.5 * (v - 1.4) / (11.0 - 1.4),
                1.0
            )
        )

    def rfs_loss(self, pred, gt, speed, t_idx, r_bar=1.0):
        """
        pred, gt: (B, T, 2)
        speed: (B, T)
        t_idx: (T,) or (B, T)
        """
        # errors
        delta = torch.abs(pred - gt) # (B, T, 2)
        delta_lat = delta[..., 1] # (B, T, 1)
        delta_lng = delta[..., 0] # (B, T, 1)

        # thresholds
        tau_lat_raw, tau_lng_raw = self.time_thresholds(t_idx)
        scale = self.speed_scale(speed)

        tau_lat = tau_lat_raw * scale
        tau_lng = tau_lng_raw * scale

        # normalized deviation, Delta > 1 == outside acceptable error
        Delta = torch.max(
            delta_lat / tau_lat,
            delta_lng / tau_lng
        )
        
        # RFS score
        score = torch.where(
            Delta <= 1,
            torch.ones_like(Delta),
            torch.pow(0.1, Delta - 1)
        )

        # normalized loss
        loss = 1.0 - score
        return loss.mean()
    
    # ---- optimizers ----
    def configure_optimizers(self):
        # NOTE: This can be extended and tuned, LR especially will differ and have an impact.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    # ---- forward / step ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']
        
        # `past` is our input (B, 16, 6) e.g. Batch x Time x (x, y, v_x, v_y, a_x, a_y)
        # and `future` is our output (B, 20, 2) e.g. Batch x Time x (x, y)

        # create all input data that we are allowed to give to a model
        model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}

        speed = torch.norm(past[..., 2:4], dim=-1)  # (B, 16)

        #create a batch of time indices from 1 to 20
        t_idx = torch.tensor([3, 5], device=future.device).unsqueeze(0).repeat(future.size(0), 1)  # (B, 2)

        pred_future = self.forward(model_inputs)  # (B, T*2)
        pred_future = pred_future.reshape_as(future)  # (B, T, 2)

        rfs_loss = self.rfs_loss(pred_future[:, [11, 19], :], future[:, [11, 19], :] , speed[:, -1].unsqueeze(1).repeat(1,2), t_idx)  # reshape to (B, T, 2)
        ade_loss = self.ade_loss(pred_future, future)
        # TODO: improve logging both to disk and to console
        self.log_dict({
            f"{stage}_rfs_loss": rfs_loss,
            f"{stage}_ade_loss": ade_loss,
        }, prog_bar=True, logger=True)

        return rfs_loss
    
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

