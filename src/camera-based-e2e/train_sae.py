import argparse
from datetime import datetime
import itertools
import math
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import SimpleProfiler

from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader, TensorDataset



# Replace with your model defined in models/
from models.base_model import LitModel, collate_with_images
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures
from models.sae import SparseAutoencoder
from loader import WaymoE2E

def normalize(t, mean, std):
    return (t - mean) / std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--latent_multiplier", type=int, default=8)
    parser.add_argument("--lambda_l1", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nw", type=int, default=0, help="number of workers for dataloader")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_check = torch.load(args.train_dataset, map_location="cpu")
    train_tok = train_check["planner_query_tok"].float()
    val_check = torch.load(args.val_dataset, map_location="cpu")
    val_tok = val_check["planner_query_tok"].float()

    mean = train_tok.mean(dim=0)
    std = train_tok.std(dim=0).clamp_min(1e-6)

    train_data = TensorDataset(train_tok)
    val_data = TensorDataset(val_tok)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
    )
        
    # Model
    in_dim = train_tok.shape[1]
    latent_dim = args.latent_multiplier * in_dim

    model = SparseAutoencoder(
        input_dim=in_dim,
        latent_dim=latent_dim
    ).to(device)

    # standard adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = float("inf")
    best_state = None
    history = []

    mean = mean.to(device)
    std = std.to(device)

    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch[0].to(device, non_blocking=True)
            batch = normalize(batch, mean, std)
            x_hat, z = model(batch)

            recon_loss = F.mse_loss(x_hat, batch)

            optimizer.zero_grad()
            l1_loss = torch.mean(torch.abs(z)) # L1 sparsity regularization on latent space
            loss = recon_loss + args.lambda_l1 * l1_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device, non_blocking=True)
                batch = normalize(batch, mean, std)

                x_hat, z = model(batch)
                recon_loss = F.mse_loss(x_hat, batch)
                l1_loss = torch.mean(torch.abs(z))
                loss = recon_loss + args.lambda_l1 * l1_loss
                val_loss += loss.item() * batch.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        history.append((train_loss, val_loss))
        print(f"Epoch {epoch+1}/{args.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_score:
            best_score = val_loss
            best_state = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state,
                "best_score": best_score,
                "input_dim": in_dim,
                "latent_dim": latent_dim,
                "latent_multiplier": args.latent_multiplier,
                "lambda_l1": args.lambda_l1,
                "lr": args.lr,
                "seed": args.seed,
                "history": history,
                "train_tokens_path": args.train_dataset,
                "val_tokens_path": args.val_dataset,},
                out_dir / "sae_checkpoint.pt")
    torch.save({"mean": mean, "std": std}, out_dir / "sae_normalization.pt")
