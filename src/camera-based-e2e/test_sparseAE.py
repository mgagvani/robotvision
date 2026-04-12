"""
Evaluate SAE reconstruction quality and trajectory impact when patching the target layer.
"""


import argparse
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader

from loader import WaymoE2E, collate_with_images
from models.base_model import LitModel
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel
from sparseAE import SparseAE


def get_sae_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        checkpoint_obj = checkpoint_obj["state_dict"]
    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Unsupported SAE checkpoint format")
    return checkpoint_obj


def select_best_trajectory(output, future):
    pred = output["trajectory"] if isinstance(output, dict) else output
    scores = output.get("scores", None) if isinstance(output, dict) else None

    bsz = future.size(0)
    t_steps = future.size(1)
    pred = pred.view(bsz, -1, t_steps, 2)

    if scores is not None and pred.size(1) > 1:
        best_idx = scores.argmin(dim=1)
    else:
        best_idx = torch.zeros(bsz, dtype=torch.long, device=future.device)

    return pred[torch.arange(bsz, device=future.device), best_idx]


def compute_batch_ade(pred_traj, future):
    return torch.norm(pred_traj - future, dim=-1).mean(dim=-1)


@contextmanager
def patched_layer_with_sae(target_layer, sae, stats):
    def replace_with_sae(module, inputs, output):
        x = output.detach()
        x_centered = x - sae.decoder.bias
        hidden = torch.relu(sae.encoder(x_centered))
        reconstructed = sae.decoder(hidden)

        stats["recon_mse_sum"] += torch.mean((reconstructed - x) ** 2).item() * x.size(0)
        stats["l0_sum"] += (hidden > 0).float().sum(dim=-1).mean().item() * x.size(0)
        stats["hook_batches"] += x.size(0)
        return reconstructed

    handle = target_layer.register_forward_hook(replace_with_sae)
    try:
        yield
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint_path", default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt", type=str, help="Frozen target-model checkpoint")
    parser.add_argument("--sae_checkpoint_path", type=str, required=True, help="Trained SparseAE checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Waymo dataset directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split to evaluate")
    parser.add_argument("--n_items", type=int, default=50_000, help="Number of examples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--block_idx", type=int, default=3, help="Transformer block index whose mlp[2] is SAE-modeled")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu, or explicit device string")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    submodel = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=40,
        n_blocks=4,
    )
    model = LitModel.load_from_checkpoint(args.model_checkpoint_path, model=submodel)
    model = model.to(device)
    model.eval()

    target_layer = model.model.blocks[args.block_idx].mlp[2]

    sae_checkpoint = torch.load(args.sae_checkpoint_path, map_location="cpu")
    sae_state = get_sae_state_dict(sae_checkpoint)
    encoder_weight = sae_state["encoder.weight"]
    dict_size, input_dim = encoder_weight.shape

    sae = SparseAE.build_from_state_dict(
        sae_state,
        target_model=model,
        input_dim=input_dim,
        dict_size=dict_size,
        compile_sae=False,
    )
    sae = sae.to(device)
    sae.eval()

    index_file = "index_val.pkl" if args.split == "val" else "index_train.pkl"
    dataset = WaymoE2E(indexFile=index_file, data_dir=args.data_dir, n_items=args.n_items)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        pin_memory=False,
    )

    baseline_ade_sum = 0.0
    patched_ade_sum = 0.0
    delta_ade_sum = 0.0
    n_examples = 0
    patch_stats = {"recon_mse_sum": 0.0, "l0_sum": 0.0, "hook_batches": 0}

    with torch.no_grad():
        for batch in loader:
            past = batch["PAST"].to(device)
            future = batch["FUTURE"].to(device)
            intent = batch["INTENT"].to(device)
            images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
            model_inputs = {"PAST": past, "IMAGES": images, "INTENT": intent}

            baseline_output = model(model_inputs)
            baseline_traj = select_best_trajectory(baseline_output, future)
            baseline_ade = compute_batch_ade(baseline_traj, future)

            with patched_layer_with_sae(target_layer, sae, patch_stats):
                patched_output = model(model_inputs)
            patched_traj = select_best_trajectory(patched_output, future)
            patched_ade = compute_batch_ade(patched_traj, future)

            batch_size = future.size(0)
            baseline_ade_sum += baseline_ade.sum().item()
            patched_ade_sum += patched_ade.sum().item()
            delta_ade_sum += (patched_ade - baseline_ade).sum().item()
            n_examples += batch_size

    if n_examples == 0:
        raise RuntimeError("No evaluation examples were processed")

    baseline_ade_mean = baseline_ade_sum / n_examples
    patched_ade_mean = patched_ade_sum / n_examples
    delta_ade_mean = delta_ade_sum / n_examples
    recon_mse_mean = patch_stats["recon_mse_sum"] / max(patch_stats["hook_batches"], 1)
    l0_mean = patch_stats["l0_sum"] / max(patch_stats["hook_batches"], 1)

    print(f"Examples evaluated: {n_examples}")
    print(f"Baseline ADE: {baseline_ade_mean:.6f}")
    print(f"Patched ADE: {patched_ade_mean:.6f}")
    print(f"ADE increase: {delta_ade_mean:.6f}")
    print(f"SAE reconstruction MSE at hooked layer: {recon_mse_mean:.6f}")
    print(f"SAE average L0: {l0_mean:.2f}")


if __name__ == "__main__":
    main()
