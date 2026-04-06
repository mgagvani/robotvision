import argparse
from pathlib import Path

import torch

from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images
from models.feature_extractors import SAMFeatures
from models.monocular import DeepMonocularModel


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

    planner_tok = []
    past = []
    future = []
    intent = []
    names = []
    traj = []
    score = []
    control = []

    with torch.inference_mode():
        for batch in loader:
            past.append(batch["PAST"])
            future.append(batch["FUTURE"])
            intent.append(batch["INTENT"])
            names.extend(batch["NAME"])

            model_inputs = {
                "PAST": batch["PAST"].to(device, non_blocking=True),
                "IMAGES": lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device),
                "INTENT": batch["INTENT"].to(device, non_blocking=True),
            }
            out = model(model_inputs)
            planner_tok.append(out["planner_query_tok"].cpu())
            traj.append(out["trajectory"].cpu())
            score.append(out["scores"].cpu())
            control.append(out["controls"].cpu())

    final = {
        "planner_query_tok": torch.cat(planner_tok, dim=0),
        "past": torch.cat(past, dim=0),
        "future": torch.cat(future, dim=0),
        "intent": torch.cat(intent, dim=0),
        "names": names,
        "meta": {
            "checkpoint": args.checkpoint,
            "index_file": args.index_file,
            "num_samples": len(names),
        },
        "trajectory": torch.cat(traj, dim=0),
        "scores": torch.cat(score, dim=0),
        "controls": torch.cat(control, dim=0),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final, output_path)
