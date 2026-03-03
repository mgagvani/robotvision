import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional


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
    def __init__(self, model: nn.Module, lr: float, lr_vision: Optional[float] = None,
                 teacher_forcing_start: float = 1.0, teacher_forcing_end: float = 0.0):
        super(LitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr
        self.hparams.lr_vision = lr_vision
        # Support scheduled teacher forcing: linear decay from start -> end across epochs
        self.hparams.teacher_forcing_start = teacher_forcing_start
        self.hparams.teacher_forcing_end = teacher_forcing_end
        # initialize current ratio to start value
        self.hparams.teacher_forcing_ratio = float(teacher_forcing_start)

        self.example_input_array = ({
            'PAST': torch.zeros((1, 16, 6)),  # PAST
            'IMAGES': [torch.zeros((1, 3, 1280, 1920)) for _ in range(6)],  # IMAGES
            'INTENT': torch.tensor([1.0]),  # INTENT
        },)

        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.depth_loss = DepthLoss(self.device)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        try:
            max_epochs = int(self.trainer.max_epochs) if (hasattr(self, 'trainer') and self.trainer is not None) else None
        except Exception:
            max_epochs = None

        start = float(self.hparams.teacher_forcing_start)
        end = float(self.hparams.teacher_forcing_end)

        if max_epochs is None or max_epochs <= 1:
            ratio = start
        else:
            # normalize current epoch in [0, 1] across (max_epochs - 1) intervals
            progress = float(self.current_epoch) / float(max_epochs - 1)
            ratio = start + (end - start) * progress

        # clamp to [0,1]
        ratio = max(0.0, min(1.0, float(ratio)))
        self.hparams.teacher_forcing_ratio = ratio
        # log for visibility
        try:
            self.log('teacher_forcing_ratio', ratio, prog_bar=True, logger=True)
        except Exception:
            pass

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory
        """
        return torch.mean(torch.norm(pred - gt, dim=-1))
    
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
    def forward(self, x: torch.Tensor, future: Optional[torch.Tensor] = None, teacher_forcing_ratio: Optional[float] = None) -> torch.Tensor:
        # Forward supports passing teacher forcing args through to the model if available
        try:
            return self.model(x, future=future, teacher_forcing_ratio=teacher_forcing_ratio)
        except TypeError:
            # Fall back if underlying model doesn't accept those args
            return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']
        
        # `past` is our input (B, 16, 6) e.g. Batch x Time x (x, y, v_x, v_y, a_x, a_y)
        # and `future` is our output (B, 20, 2) e.g. Batch x Time x (x, y)

        # create all input data that we are allowed to give to a model
        model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}

        # During training, enable teacher forcing by passing ground-truth future
        if stage == "train":
            pred_future = self.forward(model_inputs, future=future, teacher_forcing_ratio=self.hparams.teacher_forcing_ratio)
        else:
            pred_future = self.forward(model_inputs)
        pred_depth = None
        if isinstance(pred_future, dict):
            pred_future, pred_depth = pred_future["trajectory"], pred_future.get("depth", None)

        t_steps = future.shape[1]
        k_modes = self.model.n_proposals if hasattr(self.model, "n_proposals") else 1
        pred = pred_future.view(pred_future.size(0), k_modes, t_steps, 2)
        dist = torch.norm(pred - future[:, None, :, :], dim=-1)  # (B, K, T)
        ade_per_mode = dist.mean(dim=-1)  # (B, K)
        top_m = min(5, ade_per_mode.size(1))


        # Depth Loss
        if pred_depth is not None:
            front_img = images[1]  # front camera
            depth_in = F.interpolate(front_img, size=(128, 128), mode='nearest')
            loss_depth = self.depth_loss(depth_in, pred_depth, loss_fn=F.l1_loss)
        else:
            loss_depth = torch.tensor(0.0, device=self.device)
        loss_ade = ade_per_mode[:, :top_m].min(dim=1)[0].mean()
        loss_depth *= 0.1 # disabled
        loss_ade *= 1.0 # TODO: tune loss terms

        # TODO: improve logging both to disk and to console
        self.log_dict({
            f"{stage}_loss_ade": loss_ade,
            f"{stage}_loss_depth": loss_depth,
            f"{stage}_loss": loss_ade + loss_depth,
        }, prog_bar=True, logger=True)

        return loss_ade + loss_depth
    
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


