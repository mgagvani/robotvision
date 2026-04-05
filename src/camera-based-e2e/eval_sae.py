import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.base_model import LitModel, collate_with_images
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures
from models.initial_sae import SparseAutoEncoder
from loader import WaymoE2E

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str,
                    default="/scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0")
parser.add_argument("--model_ckpt", type=str,
                    default="/scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e/checkpoints/camera-e2e-epoch=04-val_loss=2.90.ckpt")
parser.add_argument("--sae_ckpt", type=str, default="/scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e/checkpoints/checkpoint_sae.pt", help="path to SAE checkpoint")
parser.add_argument("--n_items", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--out", type=str, default=None, help="optional path to save raw results as .pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load DeepMonocularModel ---
out_dim = 20 * 2
model = DeepMonocularModel(
    feature_extractor=SAMFeatures(
        model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
    ),
    out_dim=out_dim,
    n_blocks=4,
)
lit_model = LitModel.load_from_checkpoint(args.model_ckpt, model=model, lr=1e-4)
lit_model.eval()
lit_model.freeze()
lit_model.to(device)
net = lit_model.model
net.to(device)
net.eval()

# --- Load SAE (infer dims from checkpoint) ---
sae_ckpt = torch.load(args.sae_ckpt, map_location=device)
sd = sae_ckpt["model"]
in_dims = sd["encoder.weight"].shape[1]
expansion = sd["encoder.weight"].shape[0] // in_dims
sae = SparseAutoEncoder(in_dims=in_dims, expansion=expansion).to(device)
sae.load_state_dict(sd)
sae.eval()
print(f"SAE: in_dims={in_dims}, expansion={expansion}, hidden={in_dims * expansion}")

# --- Data ---
dataset = WaymoE2E(indexFile="index_val.pkl", data_dir=args.data_dir, n_items=args.n_items)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=0,
    collate_fn=collate_with_images,
    pin_memory=False,
)


def pick_best(out, B):
    """Select scorer's top-1 trajectory for each sample. Returns (B, 20, 2)."""
    traj = out["trajectory"]           # (B, K*T*2) flat
    scores = out["scores"]             # (B, K)
    K = scores.shape[1]
    T = 20
    traj_bkt2 = traj.view(B, K, T, 2)
    best_idx = scores.argmin(dim=1)    # (B,)
    return traj_bkt2[torch.arange(B, device=traj.device), best_idx]  # (B, T, 2)


def oracle_min_ade(out, future, B):
    """Min ADE across all K proposals."""
    traj = out["trajectory"]
    K = out["scores"].shape[1]
    T = 20
    traj_bkt2 = traj.view(B, K, T, 2)                              # (B, K, T, 2)
    dist = torch.norm(traj_bkt2 - future[:, None, :, :], dim=-1)   # (B, K, T)
    ade_per_mode = dist.mean(dim=-1)                                # (B, K)
    return ade_per_mode.min(dim=1).values                           # (B,)


results = {"base": {"ade": [], "fde": [], "min_ade": []},
           "sae":  {"ade": [], "fde": [], "min_ade": []}}

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        B = batch["PAST"].shape[0]
        batch["PAST"] = batch["PAST"].to(device)
        batch["INTENT"] = batch["INTENT"].to(device)
        future = batch["FUTURE"].to(device)          # (B, 20, 2)
        batch["IMAGES"] = lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)

        model_input = {"PAST": batch["PAST"], "IMAGES": batch["IMAGES"], "INTENT": batch["INTENT"]}

        out_base = net(model_input)
        out_sae  = net(model_input, sae=sae)

        for tag, out in [("base", out_base), ("sae", out_sae)]:
            pred = pick_best(out, B)                                         # (B, 20, 2)
            dist = torch.norm(pred - future, dim=-1)                         # (B, 20)
            results[tag]["ade"].append(dist.mean(dim=-1).cpu())              # (B,)
            results[tag]["fde"].append(dist[:, -1].cpu())                    # (B,)
            results[tag]["min_ade"].append(oracle_min_ade(out, future, B).cpu())  # (B,)

# --- Aggregate ---
def summarise(vals):
    v = torch.cat(vals).numpy()
    return {"mean": float(np.mean(v)), "std": float(np.std(v)),
            "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99))}

stats = {tag: {k: summarise(v) for k, v in res.items()} for tag, res in results.items()}

print(f"\n{'Metric':<20} {'Baseline':>12} {'SAE':>12} {'Delta':>12} {'% Error':>10}")
print("-" * 70)
for metric in ("ade", "fde", "min_ade"):
    b = stats["base"][metric]["mean"]
    s = stats["sae"][metric]["mean"]
    pct = (s - b) / b * 100 if b != 0 else float("nan")
    print(f"{metric:<20} {b:>12.4f} {s:>12.4f} {s - b:>+12.4f} {pct:>+9.2f}%")

print(f"\nFull stats:")
for tag in ("base", "sae"):
    print(f"\n  [{tag}]")
    for metric in ("ade", "fde", "min_ade"):
        st = stats[tag][metric]
        print(f"    {metric:<10}  mean={st['mean']:.4f}  std={st['std']:.4f}  p50={st['p50']:.4f}  p99={st['p99']:.4f}")

if args.out:
    torch.save({"stats": stats, "raw": {t: {k: torch.cat(v) for k, v in r.items()} for t, r in results.items()}}, args.out)
    print(f"\nSaved to {args.out}")
