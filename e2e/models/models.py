import pytorch_lightning as pl


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from transformers import get_cosine_schedule_with_warmup

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

        self.example_input_array = ({
            'past': torch.zeros((1, 16, 6)),  # PAST
            'images': [torch.zeros((1, 3, 1280, 1920)) for _ in range(model.num_cams)],  # IMAGES
            'intent': torch.tensor([1.0]),  # INTENT
        },)

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory
        """
        error = pred - gt  # (B, T, 2)
        # Clamp to 1e-6 instead of zero to prevent NaNs in backward pass
        loss = torch.sqrt(torch.clamp(torch.sum(error ** 2, dim=-1), min=1e-6))  # (B, T)
        return torch.mean(loss)
    
    
    # ---- optimizers ----
    def configure_optimizers(self):
        # NOTE: This can be extended and tuned, LR especially will differ and have an impact.
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-2
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    # ---- forward / step ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past = batch['past']
        future = batch['future']
        images = batch['images']
        intent = batch['intent']

        inputs = {
            'past': past,
            'images': images,
            'intent': intent
        }

        pred_future = self.forward(inputs)
        # loss = self.ade_loss(pred_future.view_as(future), future)
        loss = F.smooth_l1_loss(pred_future.view_as(future), future)

        self.log_dict({
            f"{stage}_loss": loss,
        }, prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")



class MonocularModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feature_extractor: nn.Module,
        num_cams: int = 0,
    ):
        # out_dim: (B, 40) which gets reshaped to (B, 20, 2) later
        super(MonocularModel, self).__init__()
        self.features = feature_extractor

        self.num_cams = num_cams

        # attention 
        self.feature_dim = sum(self.features.dims)  # works for both DINO and SAM
        self.key_projection = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim) # project into "key" space
        self.value_projection = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim)

        # condition the query on intent (B,) and past (B, 16, 6)
        query_input_dim = 3 + 16 * 6  # one hot -- concat -- flattened
        self.query = nn.Sequential(
            nn.Linear(query_input_dim, self.feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # learnable positional encoding
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)

        self.n_tokens = self.n_tokens * self.num_cams

        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02)) # (1, N, C)

        # MLP at end rather than directly using softmax as final output
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, out_dim),
        )


    def forward(self, x: dict) -> torch.Tensor:
        # past: (B, 16, 6), intent: int
        past, images, intent = x['past'], x['images'], x['intent']
        
        # Ref: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50%20%20order%20=%20[2,%201,%203]
        # front_cam = images[1]
        # with torch.no_grad():
        #     feats = self.features(front_cam)  # list or tensor

        # images can arrive as a tensor (B, cams, C, H, W) or list of length cams with tensors (B, C, H, W).
        if images is None:
            cams = []
        elif torch.is_tensor(images):
            assert images.dim() == 5, "Expected images tensor of shape (B, cams, C, H, W)"
            cams = [images[:, i] for i in range(images.size(1))]
        else:
            cams = list(images)

        cam_tokens = []
        with torch.no_grad():
            for cam_img in cams:
                feats = self.features(cam_img)

                if isinstance(feats, (list, tuple)):
                    tokens = torch.cat([f.flatten(2) for f in feats], dim=1)  # (B, C_total, N)
                else:
                    tokens = feats.flatten(2)  # (B, C, N)

                tokens = torch.permute(tokens, (0, 2, 1))  # (B, N, C)
                cam_tokens.append(tokens)

        intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()  # (B, 3). minus 1 --> 0, 1, 2
        past_flat = past.view(past.size(0), -1)  # (B, 96)
        query = self.query(torch.cat([intent_onehot, past_flat], dim=1)).unsqueeze(1)  # (B, 1, feature_dim)

        if len(cam_tokens) == 0:
            # No cameras: skip attention entirely, rely on intent + past only.
            out = self.decoder(query.squeeze(1))
            return out

        tokens = torch.cat(cam_tokens, dim=1)  # (B, cams * N, C)

        tokens = tokens + self.positional_encoding.to(tokens.device)
        
        # Normalize tokens to reduce overflow risk in attention logits
        tokens = F.layer_norm(tokens, tokens.shape[-1:])

        # attention
        key = self.key_projection(tokens) # (B, N, C)
        value = self.value_projection(tokens) # (B, N, C)

        # Normalize query and key to prevent NaN in attention scores with multiple cameras
        query = F.layer_norm(query, query.shape[-1:])
        key = F.layer_norm(key, key.shape[-1:])

        scores = (query @ key.permute((0, 2, 1)))  # (B, 1, N)
        scores = scores / sqrt(key.shape[2])
        scores = torch.clamp(scores, min=-100.0, max=100.0)
        scores = scores - scores.max(dim=2, keepdim=True).values
        attn = F.softmax(scores, dim=2)
        attention = attn @ value  # (B, 1, C)
        out = self.decoder(attention.squeeze(1))  # (B, 40)

        return out