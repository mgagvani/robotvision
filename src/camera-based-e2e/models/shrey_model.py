from typing import List, Dict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import disable
from pathlib import Path

import pytorch_lightning as pl
import timm

from math import sqrt


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        test_out_path: str = "test_metrics.json",
        loss_out_path: str = "scene_loss.json",
    ):
        super().__init__()
        self.model = model
        # Save config only (not the full nn.Module)
        self.save_hyperparameters(ignore=["model"])

        # test accumulators (initialized in on_test_epoch_start)
        self._test_loss_sum = None
        self._test_count = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return self.model(x)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch["PAST"] = batch["PAST"].to(device)
        batch["FUTURE"] = batch["FUTURE"].to(device)
        batch["INTENT"] = batch["INTENT"].to(device)
        batch["IMAGES"] = [img.to(device, non_blocking=True) for img in batch["IMAGES"]]
        return batch

    @staticmethod
    def per_example_ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # (B, 20, 2) -> (B,)
        return torch.norm(pred - gt, dim=-1).mean(dim=1)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.per_example_ade(pred, batch["FUTURE"]).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.per_example_ade(pred, batch["FUTURE"]).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # ---------------- TESTING ----------------

    def on_test_epoch_start(self):
        self._test_loss_sum = 0.0
        self._test_count = 0

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        per_ex = self.per_example_ade(pred, batch["FUTURE"])  # (B,)

        # accumulate correct global mean: sum over examples / count
        self._test_loss_sum += per_ex.sum().detach().cpu().item()
        self._test_count += int(per_ex.numel())

        # optional live logging (batch mean)
        self.log("test_loss", per_ex.mean(), prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        if self._test_count == 0:
            return

        mean_test = self._test_loss_sum / self._test_count

        # write once (safe for future DDP use)
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        out_path = Path(self.hparams.test_out_path)
        out_path.write_text(json.dumps(
            {"test_loss_mean": mean_test, "num_examples": self._test_count},
            indent=2
        ))

        print(f"[TEST] mean test_loss={mean_test:.6f} over {self._test_count} examples → {out_path}")

    @torch.no_grad()
    def export_scene_loss_json(self, dataloader, out_path: str | None = None) -> None:
        self.eval()  # puts LitModel + submodules in eval
        device = next(self.model.parameters()).device  # robust w/o Trainer

        loss_map: Dict[str, float] = {}
        for batch in dataloader:
            batch = self.transfer_batch_to_device(batch, device, dataloader_idx=0)
            pred = self(batch)  # (B, 20, 2)

            per_ex = self.per_example_ade(pred, batch["FUTURE"]).detach().cpu().tolist()
            ids = batch["NAME"]

            for sid, l in zip(ids, per_ex):
                loss_map[sid] = float(l)

        path = out_path or self.hparams.loss_out_path
        with open(path, "w") as f:
            json.dump(loss_map, f, indent=2)

        print(f"[TEST] Saved {len(loss_map)} scene losses → {path}")


class SAMFeatures(nn.Module):
    def __init__(
        self,
        model_name: str = "timm/sam2_hiera_tiny.fb_r896_2pt1",
        frozen: bool = True,
        feature_stage: int = -1,
    ):
        super().__init__()
        self.sam_model = timm.create_model(model_name, pretrained=True, features_only=True)
        data_config = timm.data.resolve_data_config(model=self.sam_model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        if frozen:
            for p in self.sam_model.parameters():
                p.requires_grad = False

        info = self.sam_model.feature_info
        self.feature_stage = feature_stage
        self.dims = [info.channels()[feature_stage]]
        self.patch_size = info.reduction()[feature_stage]

    @disable
    def forward(self, x):
        x_t = self.transforms(x.float())
        feats = self.sam_model(x_t)
        return [feats[self.feature_stage]]


class NewModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.features = SAMFeatures()
        self.feature_dim = sum(self.features.dims)

        self.num_queries = 8
        self.query = nn.Parameter(torch.randn(1, self.num_queries, self.feature_dim))

        self.key_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

        self.intent_vector = nn.Embedding(4, self.feature_dim)
        self.past_proj = nn.Linear(16 * 6, self.feature_dim)

    def forward(self, x):
        past, images, intent = x["PAST"], x["IMAGES"], x["INTENT"]

        # REMOVE THIS LINE (it breaks tracing)
        # print("intent values:", intent.unique())

        cam_images = [images[i] for i in [1, 2, 3]]
        larger_tensor = torch.cat(cam_images, dim=0)

        cam_feats = self.features(larger_tensor)[0]
        cam_feats = cam_feats.flatten(2).permute(0, 2, 1)

        B = past.size(0)
        T, D = cam_feats.size(1), cam_feats.size(2)

        feats = cam_feats.view(3, B, T, D)
        feats = feats.permute(1, 0, 2, 3).reshape(B, T * 3, D)

        past_token = self.past_proj(past.flatten(1)).unsqueeze(1)
        intent_token = self.intent_vector(intent.long()).unsqueeze(1)

        feats = torch.cat([past_token, intent_token, feats], dim=1)

        key = self.key_projection(feats)
        value = self.value_projection(feats)

        query = self.query.expand(B, -1, -1)
        scores = torch.matmul(query, key.transpose(1, 2))
        attn = F.softmax(scores / sqrt(self.feature_dim), dim=-1)

        out = torch.matmul(attn, value).mean(dim=1)
        return out.view(B, 20, 2)
