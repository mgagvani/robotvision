import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE_TO_KEY = {
    "score": "scores_input",
    "traj": "trajectory_feat",
    "control": "control_pred",
}


class SparseAutoEncoder(nn.Module):
    def __init__(self, in_dims: int, expansion: int, sparsity: float = 1e-4):
        super().__init__()
        self.sparsity = sparsity
        self.pre_bias = nn.Parameter(torch.zeros(in_dims))
        self.encoder = nn.Linear(in_dims, in_dims * expansion, bias=True)
        self.decoder = nn.Linear(in_dims * expansion, in_dims, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.pre_bias
        return F.relu(self.encoder(x_centered))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        x_hat = self.decode(z)
        return {"reconstruction": x_hat, "latents": z}


def load_torch_file(path: str, map_location):
    """Load a torch file while preferring weights_only for plain tensor/state_dict files."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_activation_data(path: str, mode: str) -> tuple[torch.Tensor, str]:
    """Load and flatten extracted activations for SAE training.

    Expected shapes:
      score:   scores_input      (N, K, 2C) -> (N*K, 2C)
      traj:    trajectory_feat   (N, K, C)  -> (N*K, C)
      control: control_pred      (N, K, T, 2) -> (N*K, T*2)
    """
    raw = load_torch_file(path, map_location="cpu")

    if isinstance(raw, dict):
        key = MODE_TO_KEY[mode]
        if key not in raw:
            available = ", ".join(sorted(raw.keys()))
            raise KeyError(f"Missing key '{key}' for mode '{mode}'. Available keys: {available}")
        data = raw[key]
    else:
        if mode != "score":
            raise ValueError(
                "Non-dict data is only supported for legacy score/query training. "
                "Use an extracted .pt dict for --mode traj or --mode control."
            )
        key = "legacy_query"
        data = torch.cat(raw, dim=0).squeeze(1)

    if not torch.is_tensor(data):
        raise TypeError(f"Expected tensor for {key}, got {type(data).__name__}")

    data = data.float()
    if mode in ("score", "traj"):
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])
        elif data.ndim != 2:
            raise ValueError(f"Expected {key} to have shape (N, K, C) or (N, C), got {tuple(data.shape)}")
    elif mode == "control":
        if data.ndim != 4:
            raise ValueError(f"Expected {key} to have shape (N, K, T, 2), got {tuple(data.shape)}")
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2] * data.shape[3])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return data.contiguous(), key


def load_training_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    ckpt = load_torch_file(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint '{path}' must be a dict, got {type(ckpt).__name__}")
    if "model" not in ckpt:
        available = ", ".join(sorted(ckpt.keys()))
        raise KeyError(
            f"Checkpoint '{path}' is missing key 'model'. Available keys: {available}. "
            "This usually means --ckpt points to an activation data file instead of an SAE checkpoint."
        )
    if "optimizer" not in ckpt:
        available = ", ".join(sorted(ckpt.keys()))
        raise KeyError(f"Checkpoint '{path}' is missing key 'optimizer'. Available keys: {available}")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to extracted planner activation .pt")
    parser.add_argument("--mode", type=str, choices=sorted(MODE_TO_KEY.keys()), default="score",
                        help="activation type to train on: score=scores_input, traj=trajectory_feat, control=control_pred")
    parser.add_argument("--ckpt", type=str, default="/scratch/gilbreth/chang899/codes/int/src/camera-based-e2e/camera-e2e-epoch=04-val_loss=2.90.ckpt", help="checkpoint save path")
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=250000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if os.path.abspath(args.ckpt) == os.path.abspath(args.data):
        raise ValueError(
            f"--ckpt and --data point to the same file: '{args.ckpt}'. "
            "Use a separate checkpoint path so training does not overwrite the activation data."
        )

    data, input_key = load_activation_data(args.data, args.mode)
    in_dims = data.shape[1]
    print(f"Loaded {data.shape[0]} {args.mode} examples from '{input_key}' of dim {in_dims}")

    model = SparseAutoEncoder(in_dims=in_dims, expansion=args.expansion, sparsity=args.sparsity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # init pre_bias to data mean so encoder sees zero-centered inputs from step 1
    with torch.no_grad():
        model.pre_bias.copy_(data.mean(dim=0).to(device))

    start_epoch = 0
    if os.path.exists(args.ckpt):
        start_epoch = load_training_checkpoint(args.ckpt, model, optimizer)
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
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "mode": args.mode,
            "input_key": input_key,
            "in_dims": in_dims,
            "expansion": args.expansion,
            "sparsity": args.sparsity,
        },
        args.ckpt,
    )
