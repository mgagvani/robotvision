import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SparseAutoEncoder(nn.Module):
    def __init__(self, in_dims: int, expansion: int, sparsity: float = 1e-4):
        super().__init__()
        self.sparsity = sparsity
        self.pre_bias = nn.Parameter(torch.zeros(in_dims))
        self.encoder = nn.Linear(in_dims, in_dims * expansion, bias=True)
        self.decoder = nn.Linear(in_dims * expansion, in_dims, bias=True)

    def forward(self, x: torch.Tensor) -> dict:
        x_centered = x - self.pre_bias
        z = F.relu(self.encoder(x_centered))
        x_hat = self.decoder(z) + self.pre_bias
        return {"reconstruction": x_hat, "latents": z}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to query_data.pt")
    parser.add_argument("--ckpt", type=str, default="/scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e/checkpoints/checkpoint_sae.pt", help="checkpoint save path")
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=250000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # query_data.pt is a list of (B, 1, C) tensors — one per batch from extract_queries.py
    raw = torch.load(args.data, map_location="cpu")
    data = torch.cat(raw, dim=0).squeeze(1).float()  # (N, C)
    in_dims = data.shape[1]
    print(f"Loaded {data.shape[0]} queries of dim {in_dims}")

    model = SparseAutoEncoder(in_dims=in_dims, expansion=args.expansion, sparsity=args.sparsity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # init pre_bias to data mean so encoder sees zero-centered inputs from step 1
    with torch.no_grad():
        model.pre_bias.copy_(data.mean(dim=0).to(device))

    start_epoch = 0
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    N = data.shape[0]
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        idx = torch.randint(0, N, (args.batch_size,))
        x = data[idx].to(device)

        out = model(x)
        loss_recon = F.mse_loss(out["reconstruction"], x)
        loss_l1 = args.sparsity * out["latents"].abs().sum(dim=-1).mean()
        loss = loss_recon + loss_l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keep decoder columns unit-norm so latent scale stays meaningful
        with torch.no_grad():
            model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  recon={loss_recon.item():.6f}  l1={loss_l1.item():.6f}")

    print("Training complete.")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs}, args.ckpt)
