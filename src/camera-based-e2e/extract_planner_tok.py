import argparse
from pathlib import Path

import torch

from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel

ACTIVATION_KEYS = {
    "score": ("scores_input",),
    "traj": ("trajectory_feat",),
    "control": ("control_pred",),
}

def load_model(checkpoint_path: str, device: torch.device) -> tuple[DeepMonocularModel, LitModel]:
    out_dim = 20 * 2
    model = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=out_dim,
        n_blocks=4,
        n_proposals=50,
    )
    lit_model = LitModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        lr=1e-4,
        map_location="cpu",
        weights_only=False,
    )
    model = lit_model.model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, lit_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n_items", type=int, default=None)
    parser.add_argument("--nw", type=int, default=0, help="number of workers for dataloader")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--activation",
        "--mode",
        dest="activation",
        type=str,
        choices=sorted(set(ACTIVATION_KEYS)),
        default="all",
        help="which activation target to save: score, traj, control, or all",
    )    
    args = parser.parse_args()


    device = torch.device(args.device)

    dataset = WaymoE2E(
        indexFile=args.index_file,
        data_dir=args.data_dir,
        n_items=args.n_items,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.nw,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    model, lit_model = load_model(args.checkpoint, device)

    tensors = {
        "past": [],
        "future": [],
        "intent": [],
        "trajectory_predicted": [],
        "scores_predicted": [],
    }
    for key in ACTIVATION_KEYS[args.activation]:
        tensors[key] = []
    names = []

    with torch.inference_mode():
        for batch in loader:
            tensors["past"].append(batch["PAST"])
            tensors["future"].append(batch["FUTURE"])
            tensors["intent"].append(batch["INTENT"])
            names.extend(batch["NAME"])

            model_inputs = {
                "PAST": batch["PAST"].to(device, non_blocking=True),
                "IMAGES": lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device),
                "INTENT": batch["INTENT"].to(device, non_blocking=True),
            }
            out = model(model_inputs)
            tensors["trajectory_predicted"].append(out["trajectory_predicted"].cpu())
            tensors["scores_predicted"].append(out["scores_predicted"].cpu())
            for key in ACTIVATION_KEYS[args.activation]:
                tensors[key].append(out[key].cpu())


    final = {key: torch.cat(values, dim=0) for key, values in tensors.items()}
    final.update(
        {
            "names": names,
            "meta": {
                "checkpoint": args.checkpoint,
                "index_file": args.index_file,
                "num_samples": len(names),
                "activation": args.activation,
                "saved_activation_keys": list(ACTIVATION_KEYS[args.activation]),
            },
        }
    )


    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final, output_path)
    print(f"Saved {args.activation} extraction with keys {sorted(final.keys())} to {output_path}")