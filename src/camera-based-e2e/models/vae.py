from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from math import sqrt

class VAEModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super(VAEModel, self).__init__()
        self.model = model
        self.hparams.lr = lr

        self.example_input_array = torch.zeros((1, 20, 2))

    # ---- Metrics ----
    def loss(self, recon_x, x, mu, logvar):
        # MSE for reconstruction quality
        MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
        
        # KL Divergence for latent space regularization
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return MSE + KLD
    
    # ---- optimizers ----
    def configure_optimizers(self):
        # NOTE: This can be extended and tuned, LR especially will differ and have an impact.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    # ---- forward / step ----
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        past, future, images, intent = batch['PAST'], batch['FUTURE'], batch['IMAGES'], batch['INTENT']
        
        # `past` is our input (B, 16, 6) e.g. Batch x Time x (x, y, v_x, v_y, a_x, a_y)
        # and `future` is our output (B, 20, 2) e.g. Batch x Time x (x, y)

        # create all input data that we are allowed to give to a model
        model_inputs = future
        # future[:, :, 0] /= 100
        # future[:, :, 1] /= 10

        pred_future, mu, logvar = self.forward(model_inputs)  # (B, T*2)
        loss = self.loss(pred_future.reshape_as(future), future, mu, logvar)  # reshape to (B, T, 2

        # TODO: improve logging both to disk and to console
        self.log_dict({
            f"{stage}_loss": loss,
        }, prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

class VAE_Est(nn.Module):
    def __init__(self, input_dim=40, hidden_dims=[512, 256, 128, 64, 20], latent_dim=6):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # --- ENCODER ---
        encoder_layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            prev_dim = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # --- DECODER ---
        decoder_layers = []
        # Reverse hidden dims for symmetry
        reversed_hidden = hidden_dims[::-1] + [input_dim]
        prev_dim = latent_dim
        
        for h_dim in reversed_hidden:
            decoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            prev_dim = h_dim
            
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.final_layer = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            logvar = torch.clamp(logvar, min=-10, max=10) 
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        x_norm = x
        
        B = x.size(0)
        x_flat = x_norm.view(B, -1)

        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        
        recon_flat = self.decoder(z)
        recon_norm = recon_flat.view(B, 20, 2)
        
        recon = recon_norm 
             
        return recon, mu, logvar
    
class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, latent_dim=6, num_layers=3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = 20

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Final projection: Hidden -> Coordinate (x, y)
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        
        h_flat = h_n[-1] 
        
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # 1. Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        decoder_out, _ = self.decoder_lstm(z_repeated)
        
        recon = self.final_layer(decoder_out)
        
        return recon, mu, logvar