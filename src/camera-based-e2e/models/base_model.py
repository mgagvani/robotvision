from dataclasses import asdict
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

from .losses.depth_loss import DepthLoss
from .proposal_planner import IPadConfig

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
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        lr_vision: Optional[float] = None,
        ipad_config: Optional[IPadConfig] = None,
    ):
        super(LitModel, self).__init__()
        self.model = model
        self.hparams.lr = lr
        self.hparams.lr_vision = lr_vision

        cfg = ipad_config if ipad_config is not None else IPadConfig()
        for field, value in asdict(cfg).items():
            setattr(self.hparams, field, value)

        self.example_input_array = ({
            'PAST': torch.zeros((1, 16, 6)),  # PAST
            'IMAGES': [torch.zeros((1, 3, 1280, 1920)) for _ in range(6)],  # IMAGES
            'INTENT': torch.tensor([1.0]),  # INTENT
        },)

        self.save_hyperparameters()

    # --- Data Loading ---- 
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # move actual tensors
        out = super().transfer_batch_to_device(batch, device, dataloader_idx)
        # keep encoded JPEG bytes on cpu to decode later
        if "IMAGES_JPEG" in batch:
            out["IMAGES_JPEG"] = batch["IMAGES_JPEG"]
        return out


    def decode_batch_jpeg(self, images_jpeg: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        # Flatten cameras
        flat_encoded, cam_sizes = [], []
        for cam in images_jpeg:
            cam_sizes.append(len(cam))
            flat_encoded.extend(
                jpg if isinstance(jpg, torch.Tensor) else torch.frombuffer(memoryview(jpg), dtype=torch.uint8)
                for jpg in cam
            )
        
        flat_decoded = torchvision.io.decode_jpeg(
            flat_encoded, 
            mode=torchvision.io.ImageReadMode.UNCHANGED,
            device = self.device,
        ) # list of (C, H, W) gpu tensors

        out = []
        idx = 0
        for n in cam_sizes:
            cam_list = flat_decoded[idx: idx+n]
            idx += n
            out.append(torch.stack(cam_list, dim=0))  # (B, C, H, W)
        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.depth_loss = DepthLoss(self.device)

    # ---- Metrics ----
    def ade_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Average Displacement Error -> L2 Norm -> Average Euclidean Distance between predicted and ground truth future trajectory
        """
        return torch.mean(torch.norm(pred - gt, dim=-1))

    def time_thresholds(self, t_idx):
        # Time-based thresholds at 3s and 5s.
        lat = torch.where(t_idx <= 3, 1.0, 1.8)
        lng = torch.where(t_idx <= 3, 4.0, 7.2)
        return lat, lng

    def speed_scale(self, v):
        # Speed-based scaling copied from RFS paper.
        return torch.where(
            v < 1.4,
            0.5,
            torch.where(
                v < 11.0,
                0.5 + 0.5 * (v - 1.4) / (11.0 - 1.4),
                1.0
            )
        )

    def compute_direction(self, trajectory):
        # Pad with first point so displacement stays (B, T, 2).
        padded = torch.cat([trajectory[:, :1], trajectory], dim=1)
        displacement = padded[:, 1:] - padded[:, :-1]
        lng_dir = F.normalize(displacement, p=2, dim=-1, eps=1e-6)
        lat_dir = torch.stack([-lng_dir[..., 1], lng_dir[..., 0]], dim=-1)
        return lng_dir, lat_dir

    def rfs_loss(self, pred, gt, lng_dir, lat_dir, speed, t_idx):
        """
        pred, gt: (B, T, 2)
        speed: (B,) or (B, T)
        t_idx: (T,) or (B, T)
        """
        delta = pred - gt
        delta_lng = (delta * lng_dir).sum(dim=-1).abs()
        delta_lat = (delta * lat_dir).sum(dim=-1).abs()

        tau_lat_raw, tau_lng_raw = self.time_thresholds(t_idx)
        scale = self.speed_scale(speed)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)

        tau_lat = tau_lat_raw * scale
        tau_lng = tau_lng_raw * scale

        deviation = torch.max(
            delta_lat / tau_lat,
            delta_lng / tau_lng,
        )
        score = torch.where(
            deviation <= 1,
            torch.ones_like(deviation),
            torch.pow(0.1, deviation - 1)
        )
        return (1.0 - score).mean()

    def _prepare_rfs_inputs(self, past, future, pred_future):
        speed = torch.norm(past[..., 2:4], dim=-1)[:, -1]  # (B,), speed at last observed time step
        full_lng_dir, full_lat_dir = self.compute_direction(future)
        indices = [11, 19]  # 3s and 5s into the future

        pred_slice = pred_future[:, indices, :]
        gt_slice = future[:, indices, :]
        lng_dir_slice = full_lng_dir[:, indices, :]
        lat_dir_slice = full_lat_dir[:, indices, :]
        t_idx = torch.tensor([3.0, 5.0], device=future.device).unsqueeze(0).expand(future.size(0), -1)
        return pred_slice, gt_slice, lng_dir_slice, lat_dir_slice, speed, t_idx
    
    # ---- NAVSIM-style quality target ----
    @torch.no_grad()
    def _compute_navsim_score(
        self,
        proposals: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate NAVSIM Eq. 5: S = NC * DAC * (5*EP + 5*TTC + 2*Comf) / 12

        Without agent / map data we set NC=1, DAC=1, TTC=1 and compute EP and
        Comf from trajectory geometry alone.

        Args:
            proposals: (B, K, T, 2) predicted trajectories
            future:    (B, T, 2)    ground-truth trajectory
        Returns:
            quality:   (B, K) in [0, 1]
        """
        B, K, T, _ = proposals.shape
        gt = future[:, None, :, :]  # (B, 1, T, 2)

        # --- Ego Progress (EP) ---
        gt_disp = future[:, -1] - future[:, 0]                  # (B, 2)
        gt_dist = gt_disp.norm(dim=-1, keepdim=True).clamp(min=1e-3)  # (B, 1)
        gt_dir = gt_disp / gt_dist                               # (B, 2)

        prop_disp = proposals[:, :, -1] - proposals[:, :, 0]    # (B, K, 2)
        progress = (prop_disp * gt_dir.unsqueeze(1)).sum(dim=-1) # (B, K)
        ep = (progress / gt_dist).clamp(0.0, 1.0)               # (B, K)

        # --- Comfort (Comf) ---
        dt = 0.25  # 4 Hz
        vel = (proposals[:, :, 1:] - proposals[:, :, :-1]) / dt          # (B,K,T-1,2)
        acc = (vel[:, :, 1:] - vel[:, :, :-1]) / dt                      # (B,K,T-2,2)
        jerk = (acc[:, :, 1:] - acc[:, :, :-1]) / dt                     # (B,K,T-3,2)
        jerk_mag = jerk.norm(dim=-1)                                      # (B,K,T-3)

        jerk_thresh = getattr(self.hparams, "comfort_jerk_threshold", 5.0)
        comf = (jerk_mag < jerk_thresh).float().mean(dim=-1)              # (B,K)

        nc = 1.0
        dac = 1.0
        ttc = 1.0
        quality = nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comf) / 12.0  # (B,K)

        return quality.clamp(0.0, 1.0)

    @torch.no_grad()
    def _compute_rfs_quality(
        self,
        proposals: torch.Tensor,
        reference: torch.Tensor,
        past: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-proposal RFS-style quality in [0, 1] for BCE scorer targets (iPad Eq. 4).

        Same longitudinal/lateral deviation + speed scaling as ``rfs_loss``, evaluated at
        3 s and 5 s (indices 11, 19 at 4 Hz). ``reference`` is typically batch ``FUTURE``
        (expert trajectory); it can be swapped for a route or rollout proxy when GT is absent.

        Optionally multiplies by a jerk comfort factor (same spirit as NAVSIM Comf in Eq. 5).
        """
        device = proposals.device
        indices = [11, 19]
        if reference.shape[1] <= max(indices):
            raise ValueError(
                f"reference horizon {reference.shape[1]} must exceed max RFS index {max(indices)}"
            )

        speed = torch.norm(past[..., 2:4], dim=-1)[:, -1]
        full_lng_dir, full_lat_dir = self.compute_direction(reference)

        ref_slice = reference[:, indices, :]
        lng_slice = full_lng_dir[:, indices, :]
        lat_slice = full_lat_dir[:, indices, :]

        prop_slice = proposals[:, :, indices, :]
        delta = prop_slice - ref_slice.unsqueeze(1)

        delta_lng = (delta * lng_slice.unsqueeze(1)).sum(dim=-1).abs()
        delta_lat = (delta * lat_slice.unsqueeze(1)).sum(dim=-1).abs()

        t_idx = torch.tensor([3.0, 5.0], device=device).unsqueeze(0).expand(reference.size(0), -1)
        tau_lat_raw, tau_lng_raw = self.time_thresholds(t_idx)
        scale = self.speed_scale(speed)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        tau_lat = tau_lat_raw * scale
        tau_lng = tau_lng_raw * scale

        deviation = torch.max(
            delta_lat / tau_lat[:, None, :],
            delta_lng / tau_lng[:, None, :],
        )
        score = torch.where(
            deviation <= 1,
            torch.ones_like(deviation),
            torch.pow(0.1, deviation - 1),
        )
        rfs_quality = score.mean(dim=-1)

        use_comf = getattr(self.hparams, "rfs_target_use_comfort", True)
        if use_comf:
            dt = 0.25
            vel = (proposals[:, :, 1:] - proposals[:, :, :-1]) / dt
            acc = (vel[:, :, 1:] - vel[:, :, :-1]) / dt
            jerk = (acc[:, :, 1:] - acc[:, :, :-1]) / dt
            jerk_mag = jerk.norm(dim=-1)
            jerk_thresh = getattr(self.hparams, "comfort_jerk_threshold", 5.0)
            comf = (jerk_mag < jerk_thresh).float().mean(dim=-1)
            rfs_quality = rfs_quality * comf

        return rfs_quality.clamp(0.0, 1.0)

    @torch.no_grad()
    def _compute_quality_target(
        self,
        pred: torch.Tensor,
        gt_expanded: torch.Tensor,
        future: torch.Tensor,
        tau: float,
        past: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns (B, K) quality target in [0, 1] based on score_target_type.
        """
        target_type = getattr(self.hparams, "score_target_type", "l1")
        if target_type == "navsim":
            return self._compute_navsim_score(pred, future)
        if target_type == "rfs":
            if past is None:
                raise ValueError("score_target_type='rfs' requires past trajectory")
            return self._compute_rfs_quality(pred, future, past)

        l1_target = (pred - gt_expanded).abs().sum(dim=-1).mean(dim=-1)  # (B, K)
        return torch.exp(-l1_target / max(tau, 1e-6))

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, intent = batch['PAST'], batch['FUTURE'], batch['INTENT']

        if "IMAGES" in batch:
            images = batch["IMAGES"]
        elif "IMAGES_JPEG" in batch:
            images_jpeg = batch["IMAGES_JPEG"]
            images = self.decode_batch_jpeg(images_jpeg)
        else:
            raise KeyError("Batch must contain either 'IMAGES_JPEG' or 'IMAGES' key.")

        model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}

        raw_output = self.forward(model_inputs)
        pred_depth = None
        pred_scores: torch.Tensor = None
        pred_traj_flat: torch.Tensor = None
        query_for_score: torch.Tensor = None
        proposal_list = None
        if isinstance(raw_output, dict):
            outputs = raw_output
            proposal_list = outputs.get("proposal_list", None)
            pred_scores = outputs.get("scores", None)
            pred_depth = outputs.get("depth", None)
            pred_traj_flat = outputs.get("trajectory_flat", None)
            query_for_score = outputs.get("query_for_score", None)
            pred_future = outputs["trajectory"]
        else:
            pred_future = raw_output

        pred = pred_future
        t_steps = future.shape[1]
        t2 = t_steps * 2
        k_modes = self.model.n_proposals if hasattr(self.model, "n_proposals") else 1

        if pred.ndim != 2:
            raise ValueError(f"Unexpected pred shape {pred.shape}; expected (B, T*2) or (B, {k_modes}*T*2).")

        if pred.shape[1] == t2:
            pred = pred.view(pred.size(0), 1, t_steps, 2)
        elif pred.shape[1] == k_modes * t2:
            pred = pred.view(pred.size(0), k_modes, t_steps, 2)
        else:
            raise ValueError(f"Unexpected pred shape {pred.shape}; expected (B, T*2) or (B, {k_modes}*T*2).")

        if not torch.isfinite(pred).all():
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)

        # ---- MoN L1 trajectory loss (iPad-style) ----
        # min over N proposals of mean L1 displacement per timestep
        gt = future[:, None, :, :]  # (B, 1, T, 2)

        if proposal_list is not None and len(proposal_list) > 0:
            # Discounted intermediate supervision: L = sum_k λ^(K-1-k) * MoN_L1(P_k)
            prev_w = getattr(self.hparams, "prev_weight", 0.1)
            loss_traj = torch.tensor(0.0, device=self.device)
            for proposals_k in proposal_list:
                if not torch.isfinite(proposals_k).all():
                    proposals_k = torch.nan_to_num(proposals_k, nan=0.0, posinf=1e3, neginf=-1e3)
                l1_per_mode = (proposals_k - gt).abs().sum(dim=-1).mean(dim=-1)  # (B, K)
                mon_l1 = l1_per_mode.amin(dim=1).mean()
                loss_traj = prev_w * loss_traj + mon_l1
        else:
            l1_per_mode = (pred - gt).abs().sum(dim=-1).mean(dim=-1)  # (B, K)
            loss_traj = l1_per_mode.amin(dim=1).mean()

        # ---- Metrics (for logging, not loss) ----
        refine_oracle_ades: List[torch.Tensor] = []
        with torch.no_grad():
            dist = torch.norm(pred - gt, dim=-1)  # (B, K, T) L2
            ade_per_mode = dist.mean(dim=-1)  # (B, K)
            oracle_ade = ade_per_mode.min(dim=1).values.mean()
            ade_pred = None
            if pred_scores is not None and k_modes > 1:
                pred_idx = pred_scores.detach().argmax(dim=1)
                ade_pred = ade_per_mode[torch.arange(pred.size(0), device=pred.device), pred_idx].mean()
            elif k_modes == 1:
                ade_pred = ade_per_mode.squeeze(1).mean()
            regret = (ade_pred - oracle_ade) if ade_pred is not None else None

            if proposal_list is not None and len(proposal_list) > 0:
                for proposals_k in proposal_list:
                    pk = proposals_k
                    if not torch.isfinite(pk).all():
                        pk = torch.nan_to_num(pk, nan=0.0, posinf=1e3, neginf=-1e3)
                    dist_r = torch.norm(pk - gt, dim=-1)
                    ade_pm = dist_r.mean(dim=-1)
                    refine_oracle_ades.append(ade_pm.min(dim=1).values.mean())

        # ---- RFS loss ----
        if pred_scores is not None and pred.size(1) > 1:
            rfs_pred_idx = pred_scores.detach().argmax(dim=1)
        else:
            rfs_pred_idx = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)
        pred_for_rfs = pred[torch.arange(pred.size(0), device=pred.device), rfs_pred_idx]

        pred_slice, gt_slice, lng_dir_slice, lat_dir_slice, speed, t_idx = self._prepare_rfs_inputs(
            past, future, pred_for_rfs,
        )
        rfs_unweighted = self.rfs_loss(pred_slice, gt_slice, lng_dir_slice, lat_dir_slice, speed, t_idx)
        rfs_weight = getattr(self.hparams, "rfs_weight", 0.0)
        loss_rfs = rfs_weight * rfs_unweighted

        # ---- Score loss (configurable) ----
        if k_modes > 1 and pred_scores is not None:
            if not torch.isfinite(pred_scores).all():
                pred_scores = torch.nan_to_num(pred_scores, nan=0.0, posinf=1e3, neginf=-1e3)

            score_loss_type = getattr(self.hparams, "score_loss_type", "bce")
            tau = getattr(self.hparams, "score_temperature", 5.0)

            with torch.no_grad():
                quality_target = self._compute_quality_target(pred, gt, future, tau, past=past)  # (B, K)
                best_idx = quality_target.argmax(dim=1)                               # (B,)

            if score_loss_type == "ce":
                loss_score = F.cross_entropy(pred_scores, best_idx)
            elif score_loss_type == "listnet":
                target_probs = F.softmax(quality_target / max(tau * 0.1, 1e-6), dim=1)
                loss_score = F.kl_div(
                    F.log_softmax(pred_scores, dim=1),
                    target_probs,
                    reduction="batchmean",
                )
            else:
                bce_loss = F.binary_cross_entropy_with_logits(pred_scores, quality_target)
                loss_score = bce_loss

                if score_loss_type == "bce_pairwise":
                    margin = getattr(self.hparams, "score_margin", 0.2)
                    rank_weight = getattr(self.hparams, "score_rank_weight", 0.2)
                    topk = int(getattr(self.hparams, "score_topk", 0))

                    best_scores = pred_scores.gather(1, best_idx.unsqueeze(1))  # (B,1)
                    pairwise_margin = margin - (best_scores - pred_scores)      # (B,K)
                    pairwise_margin.scatter_(1, best_idx.unsqueeze(1), 0.0)

                    if topk > 0:
                        k_eff = min(topk, pairwise_margin.size(1) - 1)
                        hardest = pairwise_margin.topk(k_eff, dim=1).values
                        rank_loss = F.relu(hardest).mean()
                    else:
                        rank_loss = F.relu(pairwise_margin).mean()
                    loss_score = bce_loss + rank_weight * rank_loss
        else:
            loss_score = torch.tensor(0.0, device=self.device)

        # Depth Loss
        if pred_depth is not None:
            front_img = images[1]
            depth_in = F.interpolate(front_img, size=(128, 128), mode='nearest')
            loss_depth = self.depth_loss(depth_in, pred_depth, loss_fn=F.l1_loss)
        else:
            loss_depth = torch.tensor(0.0, device=self.device)
        loss_depth *= 0.1

        # Score loss warmup
        sw_score = getattr(self.hparams, "score_weight", 1.0)
        warmup_epochs = getattr(self.hparams, "score_warmup_epochs", 2)
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        if current_epoch < warmup_epochs:
            effective_score_weight = 0.0
        elif current_epoch < warmup_epochs + 1:
            effective_score_weight = sw_score * (current_epoch - warmup_epochs)
        else:
            effective_score_weight = sw_score

        loss_score *= effective_score_weight
        total_loss = loss_traj + loss_depth + loss_score + loss_rfs

        # Smoothness / comfort losses on best proposal
        loss_smooth = torch.tensor(0.0, device=self.device)
        loss_collision = torch.tensor(0.0, device=self.device)
        loss_comfort = torch.tensor(0.0, device=self.device)
        sw = getattr(self.hparams, "smoothness_weight", 0.0)
        cw = getattr(self.hparams, "collision_weight", 0.0)
        cfw = getattr(self.hparams, "comfort_weight", 0.0)
        if (sw > 0 or cw > 0 or cfw > 0) and pred_for_rfs.shape[1] >= 3:
            vel = pred_for_rfs[:, 1:] - pred_for_rfs[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            jerk = acc[:, 1:] - acc[:, :-1]
            if sw > 0:
                loss_smooth = (jerk ** 2).mean()
            if cfw > 0:
                v_mid = vel[:, :-1]
                a_mid = acc
                v_speed = (v_mid ** 2).sum(dim=-1).clamp(min=0.25).sqrt()
                a_mag = (a_mid ** 2).sum(dim=-1).sqrt()
                curv = a_mag / (v_speed ** 2)
                loss_comfort = curv.mean()
        total_loss = total_loss + sw * loss_smooth + cw * loss_collision + cfw * loss_comfort

        log_payload = {
            f"{stage}_loss_traj": loss_traj,
            f"{stage}_loss_score": loss_score,
            f"{stage}_loss_depth": loss_depth,
            f"{stage}_loss_rfs": loss_rfs,
            f"{stage}_rfs_unweighted": rfs_unweighted,
            f"{stage}_loss": total_loss,
        }
        if sw > 0 or cw > 0 or cfw > 0:
            log_payload[f"{stage}_loss_smooth"] = loss_smooth
            log_payload[f"{stage}_loss_collision"] = loss_collision
            log_payload[f"{stage}_loss_comfort"] = loss_comfort
        if ade_pred is not None:
            log_payload[f"{stage}_ade_pred"] = ade_pred
            log_payload[f"{stage}_ade_oracle"] = oracle_ade
            log_payload[f"{stage}_ade_regret"] = regret
        for ri, oade in enumerate(refine_oracle_ades):
            log_payload[f"{stage}_ade_oracle_refine_{ri}"] = oade
        if pred_scores is not None and k_modes > 1:
            with torch.no_grad():
                mode_l1 = (pred - gt).abs().sum(dim=-1).mean(dim=-1)  # (B,K)
                best_idx = mode_l1.argmin(dim=1)
                pred_idx = pred_scores.argmax(dim=1)
                top1_acc = (pred_idx == best_idx).float().mean()
                s_best = pred_scores.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                masked = pred_scores.clone()
                masked.scatter_(1, best_idx.unsqueeze(1), float("-inf"))
                s_second = masked.max(dim=1).values
                log_payload[f"{stage}_score_top1_acc"] = top1_acc
                log_payload[f"{stage}_score_gap_best_second"] = (s_best - s_second).mean()
        self.log_dict(log_payload, prog_bar=True, logger=True)

        return total_loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")
    
def collate_with_images(batch):
    past = [torch.as_tensor(b["PAST"], dtype=torch.float32) for b in batch]
    future = [torch.as_tensor(b["FUTURE"], dtype=torch.float32) for b in batch]
    intent = torch.as_tensor([b["INTENT"] for b in batch])
    names = [b["NAME"] for b in batch]

    cams = list(zip(*[b["IMAGES_JPEG"] for b in batch]))  # per-camera tuples
    images_jpeg = [list(cam_imgs) for cam_imgs in cams]  # stay on CPU

    return {
        "PAST": torch.stack(past, dim=0),
        "FUTURE": torch.stack(future, dim=0),
        "INTENT": intent,
        "IMAGES_JPEG": images_jpeg,
        "NAME": names,
    }
