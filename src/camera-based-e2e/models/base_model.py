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

