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
from models.base_model import collate_with_images_tokens_depth
from models.monocular import SAMFeatures
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

def get_2d_sinusoidal_embeddings(height, width, d_model):
    if d_model % 4 != 0:
        raise ValueError("2D sinusoidal embeddings require d_model divisible by 4.")
    half_dim = d_model // 2
    y_embed = get_sinusoidal_embeddings(height, half_dim)  # (H, D/2)
    x_embed = get_sinusoidal_embeddings(width, half_dim)   # (W, D/2)
    y_grid = y_embed[:, None, :].expand(height, width, half_dim)
    x_grid = x_embed[None, :, :].expand(height, width, half_dim)
    return torch.cat([y_grid, x_grid], dim=-1).reshape(height * width, d_model)


class DiffuseLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, scheduler: DDIMScheduler, lr: float):
        super().__init__()
        self.model = model
        self.scheduler = scheduler 
        self.save_hyperparameters(ignore=['model', 'scheduler'])
        self.hparams.lr = lr
        
        # past_values = torch.load("past_normal_values.pt")
        # self.register_buffer("past_mean", past_values['mean'])
        # self.register_buffer("past_std", past_values['std'])

        # future_values = torch.load("future_normal_values.pt")
        # self.register_buffer("future_mean", future_values['mean'])
        # self.register_buffer("future_std", future_values['std'])

        self.anchors = torch.load('future_clusters_20.pt')
        self.register_buffer("past_scale", torch.tensor([160.0, 50.0, 30.0, 12.0, 1.5, 1.5])) #tuned past and future based on min-max
        self.register_buffer("future_scale", torch.tensor([160.0, 50.0]))

    # def generate_anchors(self, num):
    #     return self.anchors[torch.randperm(len(self.anchors))[:num]]
    
    # def get_closest_anchor(self, future):
    #     dist_matrix = torch.cdist(future.view(-1, 40), self.anchors.to(future.device).view(-1, 40), p=2)
    #     closest_anchor_indices = torch.argmin(dist_matrix, dim=1)
    #     return self.anchors.to(future.device)[closest_anchor_indices]

    def _shared_step(self, batch, stage):
        # print(batch.keys())
        past, future, images, tokens, intent, depths = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['TOKENS'], batch['INTENT'], batch['PRECOMPUTED_DEPTH']
        
        # Get object detection and lane detection ground truth if available
        obj_det_gt = batch.get('PRECOMPUTED_OBJ_DET', None)
        lane_gt = batch.get('PRECOMPUTED_LANE', None)

        # NORMALIZE
        # past = (past - self.past_mean) / self.past_std
        # past = torch.nan_to_num(past) #last pos is nan, as mean/std is 0
        # future_norm = (future - self.future_mean) / self.future_std 
        past = past / self.past_scale
        future_norm = future / self.future_scale

        scaler = 1

        if stage == "train":
            noise = torch.randn(future.size(0), 20, 2, device=past.device) * scaler

            # Adding noise to an anchor to generate better possible trajectories
            # noisy_anchor = self.get_closest_anchor(future).to(past.device, dtype=noise.dtype)/self.future_scale + noise
            
            bs = future_norm.shape[0]

            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bs,), device=self.device
            ).long()
            

            future_noisy = self.scheduler.add_noise(future_norm, noise, timesteps)
            
            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent, 'FUTURE': future_noisy, 'TOKENS': tokens}
            
            pred_x0, pred_depths, pred_lanes, pred_obj_dets = self.model(model_inputs, timesteps, stage)
            pred_x0 = pred_x0 * self.future_scale

            # print(depths.shape)
            # Try to convert sparse tensors to dense
            try:
                if hasattr(depths, 'is_sparse') and depths.is_sparse:
                    depths = depths.to_dense()
            except Exception as e:
                # If conversion fails, try using torch-dense method
                try:
                    depths = torch.sparse.to_dense(depths)
                except:
                    pass
            # Depth may arrive as (B, 3, 1, 1, H, W) if precompute stored a batch dim.
            if depths.dim() == 6 and depths.size(2) == 1:
                depths = depths.squeeze(2)
            elif depths.dim() == 6 and depths.size(0) == 1:
                depths = depths.squeeze(0)
            if depths.dim() == 4:
                depths = depths.unsqueeze(2)
            if depths.dim() != 5:
                raise RuntimeError(f"Unexpected depth tensor shape: {tuple(depths.shape)}")
            depths = depths.permute(1, 0, 2, 3, 4) # (3, batch, 1, 128, 128)
            # print(depths.shape, pred_depths[0].shape)
            depth_loss = sum([F.mse_loss(pred_depth, depth) for pred_depth, depth in zip(pred_depths, depths[:])])
            self.log(f"{stage}_depth_loss", depth_loss, prog_bar=True)

            # Compute lane detection loss
            if lane_gt is not None and pred_lanes is not None:
                # Try to convert sparse tensors to dense
                try:
                    if hasattr(lane_gt, 'is_sparse') and lane_gt.is_sparse:
                        lane_gt = lane_gt.to_dense()
                except Exception as e:
                    # If conversion fails, try using torch-dense method
                    try:
                        lane_gt = torch.sparse.to_dense(lane_gt)
                    except:
                        pass
                # Lane GT may arrive as (B, 3, 1, 2, H, W); drop only singleton dims.
                if lane_gt.dim() == 6 and lane_gt.size(2) == 1:
                    lane_gt = lane_gt.squeeze(2)
                elif lane_gt.dim() == 6 and lane_gt.size(0) == 1:
                    lane_gt = lane_gt.squeeze(0)
                if lane_gt.dim() != 5:
                    raise RuntimeError(f"Unexpected lane tensor shape: {tuple(lane_gt.shape)}")
                lane_gt = lane_gt.permute(1, 0, 2, 3, 4)  # (3, batch, 2, 128, 128)
                lane_loss = torch.tensor(0.0, device=past.device)
                for cam_idx, pred_lane in enumerate(pred_lanes):
                    # Precomputed lane targets are logits/probabilities with 2 classes.
                    # Use hard class labels and CE over channel dimension.
                    lane_target = lane_gt[cam_idx].argmax(dim=1).long()  # (B, H, W)
                    lane_loss = lane_loss + F.cross_entropy(pred_lane, lane_target)
                self.log(f"{stage}_lane_loss", lane_loss, prog_bar=True)
            else:
                lane_loss = torch.tensor(0.0, device=past.device)

            # Compute object detection loss
            if obj_det_gt is not None and pred_obj_dets is not None:
                obj_det_loss = self._compute_obj_det_loss(pred_obj_dets, obj_det_gt, images)
                self.log(f"{stage}_obj_det_loss", obj_det_loss, prog_bar=True)
            else:
                obj_det_loss = torch.tensor(0.0, device=past.device)

            target = future_norm * self.future_scale
            recon_loss = self.ade_loss(pred_x0, target)
            self.log(f"{stage}_recon_loss", recon_loss, prog_bar=True)

            loss = recon_loss + depth_loss * 0.8 + lane_loss * 0.8 + obj_det_loss * 0.8
            

        else:
            noise = torch.randn(past.size(0), 20, 2, device=past.device) * scaler

            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent, 'FUTURE': noise, 'TOKENS': tokens}

            pred_norm = self.model(model_inputs, None, stage)
            
            # Denormalize
            # pred = pred_norm.view(-1, 20, 2) * self.future_std + self.future_mean
            pred = pred_norm.view(-1, 20, 2) * self.future_scale
            loss = self.ade_loss(pred, future)
            
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def _normalize_xyxy_boxes(self, boxes: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """Normalize [x1, y1, x2, y2] boxes to [0, 1] and enforce valid corner ordering."""
        if img_h <= 0 or img_w <= 0:
            raise ValueError(f"Invalid image size for box normalization: {(img_h, img_w)}")

        scale = boxes.new_tensor([img_w, img_h, img_w, img_h]).view(1, 1, 4)
        boxes = boxes / scale

        x1 = torch.minimum(boxes[..., 0], boxes[..., 2]).clamp(0.0, 1.0)
        y1 = torch.minimum(boxes[..., 1], boxes[..., 3]).clamp(0.0, 1.0)
        x2 = torch.maximum(boxes[..., 0], boxes[..., 2]).clamp(0.0, 1.0)
        y2 = torch.maximum(boxes[..., 1], boxes[..., 3]).clamp(0.0, 1.0)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _compute_obj_det_loss(self, pred_obj_dets, gt_obj_dets, images):
        """Compute object detection loss between predicted and ground truth detections."""
        device = next(self.parameters()).device
        if not pred_obj_dets or 'boxes' not in gt_obj_dets or 'scores' not in gt_obj_dets or 'labels' not in gt_obj_dets:
            return torch.tensor(0.0, device=device)

        num_cameras = min(len(pred_obj_dets), gt_obj_dets['boxes'].shape[1], len(images))
        if num_cameras == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)

        for cam_idx in range(num_cameras):
            pred = pred_obj_dets[cam_idx]
            pred_scores = pred.get('pred_scores')
            pred_boxes = pred.get('pred_boxes')

            if pred_scores is None or pred_boxes is None:
                continue

            pred_scores = pred_scores.to(device=device).squeeze(-1)  # (B, K)
            pred_boxes = pred_boxes.to(device=device)  # (B, K, 4), normalized xyxy

            batch_size, top_k = pred_scores.shape
            gt_boxes = gt_obj_dets['boxes'][:, cam_idx].to(device=device, dtype=pred_boxes.dtype)
            gt_scores = gt_obj_dets['scores'][:, cam_idx].to(device=device, dtype=pred_scores.dtype)
            gt_labels = gt_obj_dets['labels'][:, cam_idx].to(device=device, dtype=torch.long)

            img_h = int(images[cam_idx].shape[-2])
            img_w = int(images[cam_idx].shape[-1])
            gt_boxes = self._normalize_xyxy_boxes(gt_boxes, img_h=img_h, img_w=img_w)

            max_det = gt_scores.shape[1]
            target_scores = torch.zeros((batch_size, top_k), device=device, dtype=pred_scores.dtype)
            target_boxes = torch.zeros((batch_size, top_k, 4), device=device, dtype=pred_boxes.dtype)
            target_labels = torch.full((batch_size, top_k), -1, device=device, dtype=torch.long)

            if max_det > 0:
                k_gt = min(top_k, max_det)
                top_gt_scores, top_gt_idx = torch.topk(gt_scores, k=k_gt, dim=1, largest=True, sorted=True)
                gather_idx = top_gt_idx.unsqueeze(-1).expand(-1, -1, 4)
                top_gt_boxes = torch.gather(gt_boxes, dim=1, index=gather_idx)
                top_gt_labels = torch.gather(gt_labels, dim=1, index=top_gt_idx)

                valid_gt = top_gt_scores > 0
                target_scores[:, :k_gt] = valid_gt.float()
                target_boxes[:, :k_gt] = top_gt_boxes
                target_labels[:, :k_gt] = torch.where(valid_gt, top_gt_labels, torch.full_like(top_gt_labels, -1))

            pos_count = target_scores.sum()
            neg_count = target_scores.numel() - pos_count

            if pos_count > 0:
                pos_weight = (neg_count / (pos_count + 1e-6)).clamp(min=1.0, max=20.0)
                score_loss = F.binary_cross_entropy_with_logits(
                    pred_scores, target_scores, pos_weight=pos_weight
                )
                positive_mask = target_scores > 0
                box_loss = F.smooth_l1_loss(pred_boxes[positive_mask], target_boxes[positive_mask], beta=0.05)

                cls_loss = torch.tensor(0.0, device=device)
                pred_logits = pred.get('pred_logits')
                if pred_logits is not None:
                    pred_logits = pred_logits.to(device=device)  # (B, K, C)
                    valid_cls_mask = positive_mask & (target_labels >= 0)
                    if valid_cls_mask.any():
                        cls_loss = F.cross_entropy(pred_logits[valid_cls_mask], target_labels[valid_cls_mask])
            else:
                score_loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores)
                box_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)

            loss = loss + score_loss + box_loss + 0.5 * cls_loss

        return loss / max(num_cameras, 1)

    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.norm(pred - gt, dim=-1))
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0.01)
    
    def forward(self, x: dict, t=None, stage="val") -> torch.Tensor:
        return self.model(x, t, stage)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

class DiffusionLTFMonocularModel(nn.Module):
    def __init__(self, feature_extractor, scheduler: DDIMScheduler, n_dims=256, n_layers=6):
        super().__init__()
        self.n_dims = n_dims
        self.scheduler = scheduler 

        # Feature extractor
        self.features = feature_extractor
        self.features.eval()
        self.feature_dim = sum(self.features.dims)
        token_h = self.features.data_config["input_size"][1] // self.features.patch_size
        token_w = self.features.data_config["input_size"][2] // self.features.patch_size
        self.n_tokens = token_h * token_w
        self.context_downsample_factor = 2
        self.context_token_h = max(1, math.ceil(token_h / self.context_downsample_factor))
        self.context_token_w = max(1, math.ceil(token_w / self.context_downsample_factor))
        self.n_context_tokens = self.context_token_h * self.context_token_w

        # need now for bicycle in forward loop
        self.register_buffer("past_scale", torch.tensor([160.0, 50.0, 30.0, 12.0, 1.5, 1.5])) #tuned past and future based on min-max
        self.register_buffer("future_scale", torch.tensor([160.0, 50.0]))


        # Context projecting/embeddings
        # self.scale_features = nn.Linear(self.feature_dim, self.n_dims)
        self.intent_embed = nn.Embedding(3, self.n_dims) # embed the different intents
        self.past_projection = nn.Linear(6, self.n_dims) # convert past waypoints to tokens
        # Condensed context tokens + explicit perception summary token per camera.
        self.n_perception_tokens = 3
        context_len = self.n_context_tokens * 3 + self.n_perception_tokens + 1 + 16
        self.positional_encoding = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((1, context_len, self.n_dims)), std=0.02)
        )

        self.context_downsample = nn.Sequential(
            nn.Conv2d(self.n_dims, self.n_dims, kernel_size=3, stride=2, padding=1, groups=self.n_dims),
            nn.Conv2d(self.n_dims, self.n_dims, kernel_size=1),
            nn.GELU(),
        )
        self.perception_token_proj = nn.Sequential(
            nn.Linear(self.n_dims + 4, self.n_dims),
            nn.LayerNorm(self.n_dims),
            nn.GELU(),
            nn.Linear(self.n_dims, self.n_dims),
        )
        self.perception_gate = nn.Sequential(
            nn.Linear(self.n_dims, self.n_dims),
            nn.Sigmoid(),
        )

        # Future Query encoder
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(self.n_dims),
            nn.Linear(self.n_dims, self.n_dims * 4),
            nn.SiLU(),
            nn.Linear(self.n_dims * 4, self.n_dims)
        )
        self.future_embeddings = get_sinusoidal_embeddings(20, self.n_dims)
        self.future_project = nn.Linear(2 + 3, self.n_dims)
        self.encoder_mlp_1 = nn.Sequential(
            nn.Linear(self.n_dims, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, self.n_dims),
            ) # should be an encoder
        self.encoder_selfattention = nn.ModuleList([
            TransformerBlock(self.n_dims, num_heads=8, mlp_dim=self.n_dims*4)
            for _ in range(2)
        ])
        self.encoder_mlp_2 = nn.Sequential(
            nn.Linear(self.n_dims, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, self.n_dims),
        )

        # CA between queries and context
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.n_dims, num_heads=8, mlp_dim=self.n_dims*4)
            for _ in range(6)
        ])

        self.visual_adapter = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.n_dims * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.n_dims * 2, self.n_dims, 3, padding=1),
        )

        self.depth_gen = nn.Sequential(
            nn.Conv2d(self.n_dims, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 1)
        )

        # Lane detection head - predicts lane line segmentation (B, 2, H, W)
        self.lane_gen = nn.Sequential(
            nn.Conv2d(self.n_dims, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 2, 1)  # 2 classes: lane vs background
        )

        # Object detection head - predicts bounding boxes from visual features
        # Uses a simple CNN-based detection head
        self.obj_det_conv = nn.Sequential(
            nn.Conv2d(self.n_dims, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
        )
        # Box regression head (4 coordinates)
        self.obj_det_box = nn.Conv2d(32, 4, 1)  # (dx, dy, w, h) per location
        # Confidence score head
        self.obj_det_score = nn.Conv2d(32, 1, 1)
        # Class logits head (driving classes from precomputed labels)
        self.obj_det_num_classes = 8
        self.obj_det_cls = nn.Conv2d(32, self.obj_det_num_classes, 1)

        # Predict output waypoints
        self.predict_waypoints = nn.Sequential(
            nn.Linear(self.n_dims, self.n_dims),
            nn.ReLU(),
            nn.Linear(self.n_dims, 2)
        )

    def unwrap_preds(self, past, control_pred, n_proposals, max_accel: float = 8.0, max_omega: float = 1.0, dt = 0.25):
        past = past * self.past_scale

        accel = torch.tanh(control_pred[..., 0]) * max_accel  # (B, K, T)
        omega: torch.Tensor = torch.tanh(control_pred[..., 1]) * max_omega  # (B, K, T)

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
        past, images, intent, future, all_tokens = x['PAST'], x['IMAGES'], x['INTENT'], x['FUTURE'], x['TOKENS']
        
        ### Get the visual tokens
        camera_tokens = []
        cams = images[0:3] # front-left, front, front-right

        depth_maps = []
        lane_maps = []
        obj_det_preds = []
        perception_tokens = []
        

        for cam, tokens in zip(cams, all_tokens):
            # with torch.no_grad():
            #     feats = self.features(cam)

            # if isinstance(feats, (list, tuple)):
            #     tokens = torch.cat([f.flatten(2) for f in feats], dim=1)
            # else:
            #     tokens = feats.flatten(2)

            # tokens = self.scale_features(torch.permute(tokens, (0, 2, 1)))
            # print(tokens.shape)
            tokens = tokens.permute(0, 2, 1) # only needed when using precomputed tokens, otherwise can directly use the features output and skip the visual adapter
            # print(tokens.shape)
            tokens = tokens.view(tokens.size(0), tokens.size(1), 32, 32)

            tokens = self.visual_adapter(tokens)

            depth_map = self.depth_gen(tokens)
            depth_maps.append(depth_map)

            # Generate lane prediction
            lane_map = self.lane_gen(tokens)
            lane_maps.append(lane_map)

            # Generate object detection predictions
            obj_det_features = self.obj_det_conv(tokens)
            pred_boxes = self.obj_det_box(obj_det_features)  # (B, 4, H, W)
            pred_scores = self.obj_det_score(obj_det_features)  # (B, 1, H, W)
            pred_logits = self.obj_det_cls(obj_det_features)  # (B, C, H, W)

            # Decode to normalized boxes [x1, y1, x2, y2] in [0, 1] with grid priors.
            B, _, H, W = pred_boxes.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=pred_boxes.device, dtype=pred_boxes.dtype),
                torch.arange(W, device=pred_boxes.device, dtype=pred_boxes.dtype),
                indexing='ij'
            )
            grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

            offset_x = torch.sigmoid(pred_boxes[:, 0])
            offset_y = torch.sigmoid(pred_boxes[:, 1])
            box_w = torch.sigmoid(pred_boxes[:, 2]) * 0.5
            box_h = torch.sigmoid(pred_boxes[:, 3]) * 0.5

            center_x = (grid_x + offset_x) / float(W)
            center_y = (grid_y + offset_y) / float(H)

            x1 = (center_x - 0.5 * box_w).clamp(0.0, 1.0)
            y1 = (center_y - 0.5 * box_h).clamp(0.0, 1.0)
            x2 = (center_x + 0.5 * box_w).clamp(0.0, 1.0)
            y2 = (center_y + 0.5 * box_h).clamp(0.0, 1.0)

            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            # Process detection predictions - flatten spatial dimensions
            pred_boxes_flat = pred_boxes.view(B, 4, -1)  # (B, 4, H*W)
            pred_scores_flat = pred_scores.view(B, 1, -1)  # (B, 1, H*W)
            pred_logits_flat = pred_logits.view(B, self.obj_det_num_classes, -1)  # (B, C, H*W)
            
            # Take top-K predictions across spatial locations
            top_k = 10  # Maximum number of detections per camera
            top_scores, top_indices = torch.topk(pred_scores_flat, k=min(top_k, pred_scores_flat.shape[2]), dim=2)
            
            # Gather corresponding boxes
            pred_boxes_gathered = torch.gather(pred_boxes_flat, 2, top_indices.expand(-1, 4, -1))
            pred_logits_gathered = torch.gather(
                pred_logits_flat,
                2,
                top_indices.expand(-1, self.obj_det_num_classes, -1)
            )
            
            # Convert to (B, N, 4) format
            obj_det_preds.append({
                'pred_boxes': pred_boxes_gathered.permute(0, 2, 1),  # (B, top_k, 4)
                'pred_scores': top_scores.permute(0, 2, 1),  # (B, top_k, 1)
                'pred_logits': pred_logits_gathered.permute(0, 2, 1),  # (B, top_k, C)
            })

            # Condense camera context tokens and build a perception summary token.
            context_map = self.context_downsample(tokens)
            context_tokens = context_map.flatten(2).permute(0, 2, 1)  # (B, 256, D) for 32x32 -> 16x16

            visual_summary = tokens.mean(dim=(2, 3))  # (B, D)
            depth_summary = depth_map.mean(dim=(2, 3))  # (B, 1)
            lane_summary = torch.softmax(lane_map, dim=1).mean(dim=(2, 3))  # (B, 2)
            det_summary = torch.sigmoid(top_scores).mean(dim=2).squeeze(1).unsqueeze(1)  # (B, 1)

            perception_descriptor = torch.cat(
                [visual_summary, depth_summary, lane_summary, det_summary], dim=1
            )  # (B, D+4)
            perception_token = self.perception_token_proj(perception_descriptor).unsqueeze(1)  # (B, 1, D)
            gate = self.perception_gate(perception_token.squeeze(1)).unsqueeze(1)  # (B, 1, D)
            context_tokens = context_tokens * (1.0 + gate)

            camera_tokens.append(context_tokens)
            perception_tokens.append(perception_token)
        
        tokens = torch.cat(camera_tokens, dim=1)
        perception_tokens = torch.cat(perception_tokens, dim=1)
         # (B, n_tokens, n_dims)
        
        ## Get intent embedding and past projection
        intent_token = self.intent_embed(intent - 1).unsqueeze(1)

        past_tokens = self.past_projection(past)

        future_context = F.one_hot(intent-1, num_classes=3).unsqueeze(1).expand(-1, 20, -1)

        context = torch.cat([tokens, perception_tokens, past_tokens, intent_token], dim=1)
        context = context + self.positional_encoding[:, :context.size(1), :]

        if stage == "train":
            queries = self.encode_future(future, future_context, t)
            
            for block in self.decoder_blocks:
                queries = block(queries, context)

            waypoints = self.predict_waypoints(queries.squeeze(1))
            waypoints = waypoints.view(-1, 20, 2)
            
            return self.unwrap_preds(past, waypoints.unsqueeze(1), 1).view(-1, 20, 2), depth_maps, lane_maps, obj_det_preds
        
        else:
            device = past.device
            
            context = context
            past_unwrap = past

            
            x_t = future.view(context.size(0), 20, 2)


            self.scheduler.set_timesteps(50, device=device)
            for t_step in self.scheduler.timesteps:
                # Predict noise or x0 (depending on sampling strategy)
                queries = self.encode_future(x_t, future_context, t_step)
                

                for block in self.decoder_blocks:
                    queries = block(queries, context)

                
                controls = self.predict_waypoints(queries)
                controls = controls.view(context.size(0), 20, 2)

                pred_original_sample = self.unwrap_preds(
                    past_unwrap, controls.unsqueeze(1), 1
                ).view(context.size(0), 20, 2)
                step_output = self.scheduler.step(model_output=pred_original_sample, timestep=t_step, sample=x_t)
                x_t = step_output.prev_sample

            

            x_t = x_t.view(past.size(0), -1, 20, 2)

            return x_t
    
    def encode_future(self, future, future_context, t):
        future = future.view(future.size(0), 20, 2)
        # print(future.shape, future_context.shape)
        # print(self.future_project(torch.cat([future, future_context], dim=-1)).shape, self.future_embeddings.to(future.device).shape, self.time_embed(t).unsqueeze(1).shape)
        future = self.future_project(torch.cat([future, future_context], dim=-1)) + self.future_embeddings.to(future.device) + self.time_embed(t).unsqueeze(1)
        future = self.encoder_mlp_1(future)

        future_atten = future
        for block in self.encoder_selfattention:
            future_atten = block(future_atten, future_atten)
        return self.encoder_mlp_2(future + future_atten)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=20, help='Number of epochs to train')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    pre_dir = args.data_dir + '/precomputed/'

    # Data 
    train_dataset = WaymoE2E(indexFile='index_train.pkl', data_dir=args.data_dir, images=True, n_items=250_000, use_precomputed=True, precomputed_dir=pre_dir+'train/')
    test_dataset = WaymoE2E(indexFile='index_val.pkl', data_dir=args.data_dir, images=True, n_items=50_000, use_precomputed=True, precomputed_dir=pre_dir+'val/')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, collate_fn=collate_with_images_tokens_depth, persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, collate_fn=collate_with_images_tokens_depth, persistent_workers=False, pin_memory=False)

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
    
    lit_model = DiffuseLitModel(model=model, scheduler=scheduler, lr=args.lr)

    base_path = Path(args.data_dir).parent.as_posix()
    trainer = pl.Trainer( #accelerator = cuda, seed_everything=42,
        # accumulate_grad_batches=2,
        max_epochs=args.max_epochs,
        logger=CSVLogger(base_path + "/logs", name=f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"),
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
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
