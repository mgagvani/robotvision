import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
    
from scipy.stats import spearmanr, kendalltau
import torch

class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float, delta: float = 1.0):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.delta = delta
        self.val_errors = []
        self.val_pred = []
        self.val_true = []

    # ---- optimizer ----
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    # ---- forward ----
    def forward(self, x):
        return self.model(x)

    # ---- shared step ----
    def _shared_step(self, batch, stage: str):
        past = batch["PAST"]
        images = batch["IMAGES"]
        intent = batch["INTENT"]
        pref_traj = batch["PREF_TRAJ"]
        pref_score = batch["PREF_SCORE"]

        pred = self(
            {
                "PAST": past,
                "IMAGES": images,
                "INTENT": intent,
                "PREF_TRAJ": pref_traj,
            }
        ).squeeze(-1)

        loss = F.smooth_l1_loss(pred, pref_score, beta=self.delta)

        self.log(
            f"{stage}_loss",
            loss,
            batch_size=past.size(0),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        if stage == "val":
            err = (pred.detach() - pref_score.detach()).abs()
            self.val_errors.append(err.cpu())
            # Save predictions and ground truth for correlation
            self.val_pred.append(pred.detach().cpu())
            self.val_true.append(pref_score.detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    # ---- Lightning 2.x hook ----
    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        import os
        import matplotlib.pyplot as plt
        import numpy as np

        if len(self.val_errors) == 0:
            print("[WARN] No validation errors collected")
            return

        # Concatenate validation results
        errors = torch.cat(self.val_errors).to(torch.float32).numpy()
        preds = torch.cat(self.val_pred).to(torch.float32).numpy()
        trues = torch.cat(self.val_true).to(torch.float32).numpy()

        self.val_errors.clear()
        self.val_pred.clear()
        self.val_true.clear()

        # Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=50)
        plt.axvline(np.mean(errors), linestyle="--", label="Mean")
        plt.xlabel("|pred − gt|")
        plt.ylabel("Count")
        plt.title(f"Validation Error Histogram (epoch {self.current_epoch})")
        plt.legend()
        out_path = os.path.join("logs", f"val_error_hist_epoch_{self.current_epoch}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] Saved histogram → {out_path}")

        # Spearman and Kendall correlations
        spearman_corr = spearmanr(trues, preds).correlation
        kendall_corr = kendalltau(trues, preds).correlation

        print(f"[INFO] Spearman correlation: {spearman_corr:.4f}")
        print(f"[INFO] Kendall tau correlation: {kendall_corr:.4f}")

        # Log to Lightning
        self.log("val_spearman", spearman_corr, prog_bar=True, logger=True)
        self.log("val_kendall", kendall_corr, prog_bar=True, logger=True)

# helps to batch data with images
def collate_with_images(batch):
    past = [torch.as_tensor(b["PAST"], dtype=torch.float32) for b in batch]
    future = [torch.as_tensor(b["FUTURE"], dtype=torch.float32) for b in batch]
    intent = torch.as_tensor([b["INTENT"] for b in batch])
    names = [b["NAME"] for b in batch]

    padded_trajs = []

    for b in batch:
        traj = torch.as_tensor(b["PREF_TRAJ"], dtype=torch.float32)
        for _ in range(len(traj), 21):
            traj = F.pad(traj, (0,0,0,1), value=0.0)  # pad to length 20
        
        padded_trajs.append(traj)

    # Extra values for preference trajectories 
    pref_traj = padded_trajs
    pref_score = torch.as_tensor([b["PREF_SCORE"] for b in batch], dtype=torch.float32)

    cams = list(zip(*[b["IMAGES"] for b in batch]))  # per-camera tuples
    images = [torch.stack(cam_imgs, dim=0) for cam_imgs in cams]  # stay on CPU

    return {
        "PAST": torch.stack(past, dim=0),
        "FUTURE": torch.stack(future, dim=0),
        "INTENT": intent,
        "IMAGES": images,
        "NAME": names,
        "PREF_TRAJ": torch.stack(pref_traj, dim=0),
        "PREF_SCORE": pref_score,
    }
