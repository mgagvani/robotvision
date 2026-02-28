from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from pathlib import Path

import numpy as np

@dataclass
class GTRSConfig:
    d_model: int = 256
    vocab_dropout: float = 0.5
    d_ffn: int = 2048
    n_head: int = 8
    n_layers: int = 4
    n_past: int = 16  # 4s @ 4Hz
    vocab_size: int = 16384

    loss_top_n: int = 64 # in DrivoR/DeepMonocular, N=50, n=5. Here N=1024, n=64
    # 'mse': attempt to regress the error of the trajectory
    # 'reinforce': minimize the expectation of error (vs. reinforce maximize exp of reward from samples), by calculating probs of errors thru softmax. includes regularization (entropy) 
    loss_type: Literal['mse', 'reinforce'] = 'reinforce'
    loss_tau_base: float = 4.0 # tau = 4.0 * (0.9 ** epoch)
    loss_tau_decay: float = 0.9
    loss_entropy_lambda: float = 0.01

class GTRSModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, *, out_dim: Optional[int]):
        super().__init__()
        self.cfg = GTRSConfig()
        self.features = feature_extractor
        self.d_features = sum(self.features.dims)
        h, w = self.features.data_config["input_size"][1:]
        self.n_img_tokens = (h // self.features.patch_size) * (w // self.features.patch_size)

        # load vocab
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(Path(__file__).parent.parent / f"vocab_{self.cfg.vocab_size}.npy"))
            , requires_grad=False
        )
        # vocab shape: (N_vocab, T, 2), flatten each entry to T*2 for the MLP
        self.vocab_dim = self.vocab[0].numel()  # e.g. 80 timesteps * 2 = 160
        self.n_proposals = self.vocab.shape[0]

        # out dim check
        if out_dim is not None and out_dim // 2 == self.vocab.shape[1]:
            raise ValueError(f"out_dim should be None or {self.vocab.shape[1] * 2}, but got {out_dim}")


        # conv 1
        self.down_conv = nn.Conv1d(self.d_features, self.cfg.d_model, 1, 1)

        # positional encoding for visual tokens
        self.positional_encoding = nn.Parameter(nn.init.trunc_normal_(torch.zeros((1, self.n_img_tokens, self.d_features)), std=0.02)) # (1, N, C)

        # vocabulary embedding: (N_vocab, vocab_dim) -> (N_vocab, d_model)
        self.vocab_embed = nn.Sequential(
            nn.Linear(self.vocab_dim, self.cfg.d_ffn),
            nn.GELU(),
            nn.Linear(self.cfg.d_ffn, self.cfg.d_model),
        )

        # big transformer to transform...
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self.cfg.d_model, self.cfg.n_head, self.cfg.d_ffn,
                dropout=0.0, batch_first=True
            ), self.cfg.n_layers
        )

        # ego status encoding
        self.status_encoding = nn.Sequential(
            nn.Linear(self.cfg.n_past * 6 + 3, self.cfg.d_ffn),
            nn.GELU(),
            nn.Linear(self.cfg.d_ffn, self.cfg.d_model),
        )

        # heads
        # for now, just predict the ADE of each trajectory as
        # a proxy for quality, same as DeepMonocularModel.
        # TODO: add extra heads for other metrics such as collision or lane keeping.
        self.heads = nn.ModuleDict({
            "scores": nn.Sequential(
                nn.Linear(self.cfg.d_model, self.cfg.d_ffn),
                nn.GELU(),
                nn.Linear(self.cfg.d_ffn, self.cfg.d_ffn), 
                nn.GELU(),
                nn.Linear(self.cfg.d_ffn, 1),
            )
        }) 

    
    def extract_tokens(self, cameras: torch.Tensor) -> torch.Tensor:
        '''
        separate function to flatten vit features
        '''
        B, N, C, H, W = cameras.shape  # (B, n_cameras, 3, H, W)
        cam_inputs = cameras.reshape(B * N, C, H, W)  
        feats_vit = self.features(cam_inputs)  # list/tuple or tensor
        if isinstance(feats_vit, (list, tuple)):
            feats_vit = torch.cat([f.flatten(2) for f in feats_vit], dim=1)  # (B, C_total, N)
        else:
            feats_vit = feats_vit.flatten(2)  # (B, C, N)
        tokens = feats_vit.permute(0, 2, 1)  
        T = tokens.shape[1]
        return tokens.reshape(B, N, T, self.d_features) # (B, n_cameras, n_tokens, d_features)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        past, images, intent = x["PAST"], x["IMAGES"], x["INTENT"]
        B = past.size(0)

        # output
        out = {}

        # ViT features
        visual_tokens = self.extract_tokens(images[1][:, None, ...]) + self.positional_encoding

        # d_features --> d_model
        visual_tokens = visual_tokens.flatten(start_dim=1, end_dim=2) # (b, n_cameras * n_tokens, d_features)
        tokens: torch.Tensor = self.down_conv(visual_tokens.transpose(1,2)).transpose(1,2) # (b, n_c * n_t, d_model)

        # If training, dropout vocab_dropout trajectories from trajectory vocabulary. 
        # GTRS shows this improves generalization
        if self.training and self.cfg.vocab_dropout:
            num_select = int(self.cfg.vocab_dropout * self.vocab.shape[0])  # e.g. 0.5 * 1024 = 512
            indices = torch.randperm(self.vocab.shape[0], device=self.vocab.device)[:num_select]
            vocab = self.vocab[indices]
        else:
            vocab = self.vocab
        out['trajectory'] = vocab.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1)

        # flatten (N_vocab, T, 2) -> (N_vocab, T*2) then embed to (N_vocab, d_model)
        vocab_flat = vocab.reshape(vocab.shape[0], -1)
        embedded_vocab = self.vocab_embed(vocab_flat)

        # (B, N_vocab, d_model) -> (B, N_vocab, d_model)
        # (target sequence, memory sequence) as input
        tr_out = self.transformer(embedded_vocab.unsqueeze(0).expand(B, -1, -1), tokens) 

        # GTRS simply adds this to transformer output
        dist_status = tr_out + self.status_encoding(torch.cat([past.reshape(B, -1), F.one_hot((intent - 1).long(), num_classes=3).float()], dim=1)).unsqueeze(1)

        for k, head in self.heads.items():
            out[k] = head(dist_status).squeeze(-1)  # (B, N_vocab)

        return out

        


