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


class SimpleLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super(SimpleLitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr

        self.example_input_array = ({
            'PAST': torch.zeros((1, 16, 6)),  # PAST
            'IMAGES': [torch.zeros((1, 3, 1280, 1920)) for _ in range(6)],  # IMAGES
            'INTENT': torch.zeros(1)  # INTENT
        },)

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory"""
        return torch.mean(torch.norm(pred - gt, dim=-1))

    # ---- optimizers ----
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    # ---- forward / step ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']

        # create all input data that we are allowed to give to a model
        model_inputs = {
            'PAST': past,
            'IMAGES': images,
            'INTENT': intent
        }

        pred_future = self.forward(model_inputs)  # (B, T*2)
        loss = self.ade_loss(pred_future.reshape_as(future), future)

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

    # per-camera tuples (list of length 6, each containing B tensors)
    cams = list(zip(*[b["IMAGES"] for b in batch]))
    images = [torch.stack(cam_imgs, dim=0) for cam_imgs in cams]  # keep on CPU

    return {
        "PAST": torch.stack(past, dim=0),
        "FUTURE": torch.stack(future, dim=0),
        "IMAGES": images,
        "INTENT": intent,
        "NAME": names,
    }