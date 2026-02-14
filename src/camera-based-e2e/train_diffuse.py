import argparse 
from datetime import datetime
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd
import random
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from diffusers import DDIMScheduler
from diffusers.models.embeddings import Timesteps
import numpy as np

from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images
from models.monocular import MonocularModel, DeepMonocularModel, SAMFeatures
from models.blocks import TransformerBlock

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        original_shape = time.shape
        time = time.flatten()
        
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        if len(original_shape) > 1:
            embeddings = embeddings.view(*original_shape, -1)
        
        return embeddings

def get_sinusoidal_embeddings(n_waypoints, d_model):
    # n_waypoints = 16, d_model = token dimension (e.g., 128)
    pe = torch.zeros(n_waypoints, d_model)
    position = torch.arange(0, n_waypoints, dtype=torch.float).unsqueeze(1)
    
    # The denominator term (10000^(2i/d_model))
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # Fill sine for even indices, cosine for odd
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe



class DiffuseLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, scheduler: DDIMScheduler, lr: float, n_traj):
        super().__init__()
        self.model = model
        self.scheduler = scheduler 
        self.save_hyperparameters(ignore=['model', 'scheduler'])
        self.hparams.lr = lr
        self.n_traj = n_traj
        
        # past_values = torch.load("past_normal_values.pt")
        # self.register_buffer("past_mean", past_values['mean'])
        # self.register_buffer("past_std", past_values['std'])

        # future_values = torch.load("future_normal_values.pt")
        # self.register_buffer("future_mean", future_values['mean'])
        # self.register_buffer("future_std", future_values['std'])

        self.anchors = torch.load('future_clusters_20.pt')
        self.register_buffer("past_scale", torch.tensor([160.0, 50.0, 30.0, 12.0, 1.5, 1.5])) #tuned past and future based on min-max
        self.register_buffer("future_scale", torch.tensor([160.0, 50.0]))

    def generate_anchors(self, num):
        return self.anchors[torch.randperm(len(self.anchors))[:num]]
    
    def get_closest_anchor(self, future):

        dist_matrix = torch.cdist(future.view(-1, 40), self.anchors.to(future.device).view(-1, 40), p=2)
        closest_anchor_indices = torch.argmin(dist_matrix, dim=1)
        return self.anchors.to(future.device)[closest_anchor_indices]

    def _shared_step(self, batch, stage):
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']

        # NORMALIZE
        # past = (past - self.past_mean) / self.past_std
        # past = torch.nan_to_num(past) #last pos is nan, as mean/std is 0
        # future_norm = (future - self.future_mean) / self.future_std 
        past = past / self.past_scale
        future_norm = future / self.future_scale

        scaler = 1

        if stage == "train":
            noise = torch.randn(future.size(0), 20, 2, device=past.device) * scaler
            noise = noise + self.get_closest_anchor(future).to(past.device, dtype=noise.dtype)/self.future_scale
            
            bs = future_norm.shape[0]

            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bs,), device=self.device
            ).long()
            

            future_noisy = self.scheduler.add_noise(future_norm, noise, timesteps)
            
            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent, 'FUTURE': future_noisy}
            
            pred_x0, score = self.model(model_inputs, timesteps, stage)

            
            target = future_norm#.view(future_norm.size(0), -1) # for x0
            # target = noise.view(noise.size(0), -1) # for noise
            # target = noise

            pred_reconstruct = pred_x0*self.future_scale
            mask = timesteps > 900
            score_loss= F.mse_loss(torch.mean(torch.norm(pred_reconstruct- future, dim=-1), dim=1)*mask, score.view(-1)*mask)
            # print(torch.norm(pred_reconstruct- future, dim=-1))
            # print(torch.mean(torch.norm(pred_reconstruct- future, dim=-1), dim=1).shape, score.view(-1))
            # print(score_loss)
            loss = F.mse_loss(pred_x0, target) + score_loss / 5000 

        else:
            anchors = self.generate_anchors(self.n_traj).to(past.device, dtype=past.dtype)/self.future_scale
            noise = torch.randn(past.size(0),self.n_traj, 20, 2, device=past.device) * scaler
            noisy_anchors = noise + anchors.to(past.device, dtype=noise.dtype)

            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent, 'FUTURE': noisy_anchors}

            pred_norm = self.model(model_inputs, None, stage)
            
            # Denormalize
            # pred = pred_norm.view(-1, 20, 2) * self.future_std + self.future_mean
            pred = pred_norm.view(-1, 20, 2) * self.future_scale
            loss = self.ade_loss(pred, future)
            
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.norm(pred - gt, dim=-1))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
    
    def forward(self, x: dict, t=None, stage="val") -> torch.Tensor:
        return self.model(x, t, stage)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

class DiffusionLTFMonocularModel(nn.Module):
    def __init__(self, feature_extractor, scheduler: DDIMScheduler, n_dims=256, n_traj=20, n_layers=6):
        super().__init__()
        self.n_dims = n_dims
        self.scheduler = scheduler 
        self.n_traj = n_traj

        # Feature extractor
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)
        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)

        # need now for bicycle in forward loop
        self.register_buffer("past_scale", torch.tensor([160.0, 50.0, 30.0, 12.0, 1.5, 1.5])) #tuned past and future based on min-max
        self.register_buffer("future_scale", torch.tensor([160.0, 50.0]))


        # Context projecting/embeddings
        self.scale_features = nn.Linear(self.feature_dim, self.n_dims)
        self.intent_embed = nn.Embedding(3, self.n_dims) # embed the different intents
        self.past_projection = nn.Linear(6, self.n_dims) # convert past waypoints to tokens
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens + 1 + 16, self.n_dims)), std=0.02))

        # Future Query encoder
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(self.n_dims),
            nn.Linear(self.n_dims, self.n_dims * 4),
            nn.SiLU(),
            nn.Linear(self.n_dims * 4, self.n_dims)
        )
        self.future_embeddings = get_sinusoidal_embeddings(20, self.n_dims)
        self.future_project = nn.Linear(2, self.n_dims)
        self.encoder_mlp_1 = nn.Sequential(
            nn.Linear(self.n_dims, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, self.n_dims),
            ) # should be an encoder
        self.encoder_selfattention = nn.ModuleList([
            TransformerBlock(self.n_dims, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(2)
        ])
        self.encoder_mlp_2 = nn.Sequential(
            nn.Linear(self.n_dims, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, self.n_dims),
        )

        # CA between queries and context
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.n_dims, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(6)
        ])

        # Scoring output from 
        self.scorer = nn.Sequential(
            nn.Linear(self.n_dims*20, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, 1)
        )

        # Predict output waypoints
        self.predict_waypoints = nn.Sequential(
            nn.Linear(self.n_dims, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, 2)
        )

    def unwrap_preds(self, past, control_pred, n_proposals, max_accel: float = 8.0, max_omega: float = 1.0, dt = 0.25):
        past = past * self.past_scale
        control_pred = control_pred * self.future_scale

        accel = torch.tanh(control_pred[..., 0]) * max_accel  # (B, K, T)
        omega = torch.tanh(control_pred[..., 1]) * max_omega  # (B, K, T)

        x_state = past[:, -1, 0].unsqueeze(1).expand(-1, n_proposals).clone()
        y_state = past[:, -1, 1].unsqueeze(1).expand(-1, n_proposals).clone()
        vx0 = past[:, -1, 2]
        vy0 = past[:, -1, 3]
        speed_state = torch.sqrt(vx0 * vx0 + vy0 * vy0 + 1e-6).unsqueeze(1).expand(-1, n_proposals).clone()
        heading_state = torch.atan2(vy0, vx0).unsqueeze(1).expand(-1, n_proposals).clone()

        xy_steps = []
        for t in range(20):
            x_state = x_state + speed_state * torch.cos(heading_state) * dt
            y_state = y_state + speed_state * torch.sin(heading_state) * dt
            xy_steps.append(torch.stack([x_state, y_state], dim=-1))

            heading_state = heading_state + omega[:, :, t] * dt
            speed_state = torch.clamp_min(speed_state + accel[:, :, t] * dt, 0.0)

        traj_xy = torch.stack(xy_steps, dim=2) / self.future_scale  # (B, K, T, 2)
        return traj_xy.reshape(traj_xy.size(0), -1)  # (B, K*T*2)

    def forward(self, x, t, stage):
        past, images, intent, future = x['PAST'], x['IMAGES'], x['INTENT'], x['FUTURE']
        
        ### Get the visual tokens
        front_cam = images[1]
        with torch.no_grad():
            feats = self.features(front_cam)

        if isinstance(feats, (list, tuple)):
            tokens = torch.cat([f.flatten(2) for f in feats], dim=1)
        else:
            tokens = feats.flatten(2)
        tokens = self.scale_features(torch.permute(tokens, (0, 2, 1)))
        
        ## Get intent embedding and past projection
        intent_token = self.intent_embed(intent - 1).unsqueeze(1)
        past_tokens = self.past_projection(past)


        context = torch.cat([tokens, past_tokens, intent_token], dim=1) + self.positional_encoding

        if stage == "train":
            queries = self.encode_future(future, t)
            
            for block in self.decoder_blocks:
                queries = block(queries, context)


            waypoints = self.predict_waypoints(queries.squeeze(1))
            score = self.scorer(queries.view(past.size(0), -1))
            
            return self.unwrap_preds(past, waypoints.unsqueeze(1), 1).view(-1, 20, 2), score
        else:
            device = past.device
            
            context = context.repeat_interleave(self.n_traj, dim=0)
            past_unwrap = past.repeat_interleave(self.n_traj, dim=0)

            
            x_t = future.view(context.size(0), 20, 2)

            save_query= None

            self.scheduler.set_timesteps(50, device=device)
            for t_step in self.scheduler.timesteps:
                # Predict noise or x0 (depending on sampling strategy)
                queries = self.encode_future(x_t, t_step)
                

                for block in self.decoder_blocks:
                    queries = block(queries, context)

                
                pred_original_sample = self.predict_waypoints(queries)
                save_query = queries
                step_output = self.scheduler.step(model_output=pred_original_sample, timestep=t_step, sample=x_t)
                x_t = step_output.prev_sample
                x_t = self.unwrap_preds(past_unwrap, x_t.unsqueeze(1), 1)
                x_t = x_t.view(context.size(0), 20, 2)
            

            scores = self.scorer(save_query.view(x_t.size(0), -1))
            scores = scores.view(past.size(0), -1, 1)
            x_t = x_t = x_t.view(past.size(0), -1, 20, 2)


            max_indices = torch.argmax(scores, dim=1)
            index_reshaped = max_indices.view(past.size(0), 1, 1, 1).expand(past.size(0), 1, 20, 2)
            predicted_x_t = torch.gather(x_t, 1, index_reshaped)

            return predicted_x_t
    
    def encode_future(self, future, t):
        future = self.future_project(future) + self.future_embeddings.to(future.device) + self.time_embed(t).unsqueeze(1)
        future = self.encoder_mlp_1(future)
        
        future_atten = future
        for block in self.encoder_selfattention:
            future_atten = block(future_atten, future_atten)
        return self.encoder_mlp_2(future + future_atten)


class DiffusionMonocularModel(nn.Module):
    def __init__(self, feature_extractor, scheduler: DDIMScheduler, out_dim, n_layers=3):
        super().__init__()
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)
        self.scheduler = scheduler 

        query_input_dim = 3 + 16 * 6 + 40 + self.feature_dim
        self.query_init = nn.Linear(query_input_dim, self.feature_dim)

        self.n_tokens = self.features.data_config["input_size"][1] // self.features.patch_size * (self.features.data_config["input_size"][2] // self.features.patch_size)
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_tokens, self.feature_dim)), std=0.02))

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.SiLU(),
            nn.Linear(self.feature_dim * 4, self.feature_dim)
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(self.feature_dim, num_heads=8, mlp_dim=self.feature_dim*4)
            for _ in range(n_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 40),
        )

    def forward(self, x, t, stage):
        past, images, intent, future = x['PAST'], x['IMAGES'], x['INTENT'], x['FUTURE']
        
        front_cam = images[1]
        with torch.no_grad():
            feats = self.features(front_cam)

        if isinstance(feats, (list, tuple)):
            tokens = torch.cat([f.flatten(2) for f in feats], dim=1)
        else:
            tokens = feats.flatten(2)
        tokens = torch.permute(tokens, (0, 2, 1)) + self.positional_encoding
        
        intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()
        past_flat = past.view(past.size(0), -1)

        if stage == "train":
            t_emb = self.time_embed(t)
            future_flat = future.view(future.size(0), -1)
            
            query = self.query_init(torch.cat([intent_onehot, past_flat, future_flat, t_emb], dim=1)).unsqueeze(1)

            for block in self.blocks:
                query = block(query, tokens)

            return self.decoder(query.squeeze(1))
        else:
            device = past.device
            x_t = torch.randn(past.size(0), 40, device=device)
            
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, device=device)

            for t_step in self.scheduler.timesteps:
                timesteps_batch = torch.full((past.size(0),), t_step, device=device, dtype=torch.long)
                t_emb = self.time_embed(timesteps_batch)
                
                # Predict noise or x0 (depending on sampling strategy)
                query = self.query_init(torch.cat([intent_onehot, past_flat, x_t, t_emb], dim=1)).unsqueeze(1)
                
                for block in self.blocks:
                    query = block(query, tokens)
                
                pred_original_sample = self.decoder(query.squeeze(1))
                
                step_output = self.scheduler.step(model_output=pred_original_sample, timestep=t_step, sample=x_t)
                x_t = step_output.prev_sample
            
            return x_t 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=4, help='Number of epochs to train')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    # Data 
    train_dataset = WaymoE2E(indexFile='index_train.pkl', data_dir=args.data_dir, images=True, n_items=250000)
    test_dataset = WaymoE2E(indexFile='index_val.pkl', data_dir=args.data_dir, images=True, n_items=500)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, collate_fn=collate_with_images, persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample", # sample or epsilon for x0 or noise
        clip_sample=False
    )

    in_dim = 16 * 6
    out_dim = 20 * 2

    model = DiffusionLTFMonocularModel(
        feature_extractor=SAMFeatures(model_name="timm/vit_pe_spatial_tiny_patch16_512.fb"), 
        scheduler=scheduler,
    )
    
    lit_model = DiffuseLitModel(model=model, scheduler=scheduler, lr=args.lr, n_traj=20)

    base_path = Path(args.data_dir).parent.as_posix()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger(base_path + "/logs", name=f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"),
        precision="bf16-mixed",
        # gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, 
                            dirpath=base_path + '/checkpoints', 
                            filename='e2e-diffuse-{epoch:02d}-{val_loss:.2f}'),
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Export loss graph
    try:
        base_path = Path(base_path)
        run_dir = sorted((base_path / "logs").glob("camera_e2e_*"))[-1] 
        metrics = pd.read_csv(run_dir / "version_0" / "metrics.csv")
        train = metrics[metrics["train_loss"].notna()]
        val = metrics[metrics["val_loss"].notna()]

        plt.figure()
        plt.plot(train["step"], train["train_loss"], label="train_loss")
        plt.plot(val["step"], val["val_loss"], label="val_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        out = Path("./visualizations")
        out.mkdir(exist_ok=True)
        plt.savefig(out / "diffuse.png", dpi=200)
    except Exception as e:
        print(f"Could not save loss plot: {e}")