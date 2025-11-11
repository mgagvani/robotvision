import argparse 
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from loader import WaymoE2E

class BaseModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BaseModel, self).__init__()
        
        # This is literally just linear regression = y_hat = Wx + b
        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
    
class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super(LitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr

        self.in_dim = model.nn[0].in_features
        self.example_input_array = torch.zeros(1, self.in_dim, dtype=torch.float32)

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch.values() # TODO: parse this robustly
        
        # For now, `past` is our input (B, 16, 6) e.g. Batch x Time x (x, y, v_x, v_y, a_x, a_y)
        # and `future` is our output (B, 16, 2) e.g. Batch x Time x (x, y)

        B, T, F = past.shape
        past = past.view(B, T * F)  # Flatten time and features
        future = future.view(B, -1)  # Flatten time and features

        pred_future = self.forward(past)  # (B, T*2)
        loss = self.ade_loss(pred_future.view(B, -1, 2), future.view(B, -1, 2))

        # TODO: improve logging both to disk and to console
        self.log_dict({
            f"{stage}_loss": loss,
        }, prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # Data 
    # TODO - make this use a proper train / val split, and to sample only specific data that is long-tailed (e.g., difficult and out of distribution)
    train_dataset = WaymoE2E(batch_size=args.batch_size, indexFile='index.pkl', data_dir=args.data_dir, images=False, n_items=1000)
    test_dataset = WaymoE2E(batch_size=args.batch_size, indexFile='index.pkl', data_dir=args.data_dir, images=False, n_items=100)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    # Model
    in_dim = 16 * 6  # Past: (B, 16, 6)
    out_dim = 20 * 2  # Future: (B, 20, 2)

    base_model = BaseModel(in_dim=in_dim, out_dim=out_dim)
    lit_model = LitModel(model=base_model, lr=args.lr)

    base_path = Path(args.data_dir).parent.as_posix()
    # We don't want to save logs or checkpoints in the home directory - it'll fill up fast
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger(base_path + "logs", name=f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"),
        callbacks=[
            ModelCheckpoint(monitor='val_loss',
                             mode='min', 
                             save_top_k=1, 
                             dirpath=base_path + '/checkpoints',
                             filename='camera-e2e-{epoch:02d}-{val_loss:.2f}'
                            ),
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Print summary of training progerss



    