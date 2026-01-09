import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from pathlib import Path

# Add parent directory to path to import original models (if needed in the future)
# Note: collate_with_images is not used here - we use create_filtered_collate_fn in train_ablation.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class AblationBaseModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AblationBaseModel, self).__init__()
        
        # This is literally just linear regression = y_hat = Wx + b
        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        
    def forward(self, x: dict) -> torch.Tensor:
        past, images, intent = x['PAST'], x['IMAGES'], x['INTENT']
        x = past.reshape(past.size(0), -1)  # Flatten to (B, feature_dim)
        return self.nn(x)
    
class AblationLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super(AblationLitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr

        # Lightweight, shape-correct example input to avoid summary-time OOM and shape mismatch
        # Handle both AblationBaseModel (has .nn) and AblationMonocularModel (has .query)
        if hasattr(self.model, 'nn'):
            # Base model
            in_features = self.model.nn[0].in_features  # Linear input dimension
            num_feats = max(1, in_features // 16)       # per-timestep features
        else:
            # Vision model - infer from query layer
            # Default to 6 features per timestep (position + velocity)
            num_feats = 6
        
        self.example_input_array = ({
            'PAST': torch.zeros((1, 16, num_feats)),                   # PAST
            'IMAGES': [torch.zeros((1, 3, 8, 8)) for _ in range(6)],   # tiny placeholder images
            'INTENT': torch.zeros((1,), dtype=torch.long),             # INTENT
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']
        
        # `past` is our input (B, 16, num_features) 
        # and `future` is our output (B, 20, 2) e.g. Batch x Time x (x, y)

        # create all input data that we are allowed to give to a model
        model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}

        pred_future = self.forward(model_inputs)  # (B, T*2)
        loss = self.ade_loss(pred_future.reshape_as(future), future)  # reshape to (B, T, 2)

        # TODO: improve logging both to disk and to console
        self.log_dict({
            f"{stage}_loss": loss,
        }, prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")


