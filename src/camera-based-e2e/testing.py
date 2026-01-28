import argparse, glob
from pathlib import Path
import torch

from loader import WaymoE2E
from models.base_model import collate_with_images
from models.shrey_model import NewModel, LitModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to .ckpt (default: newest in checkpoints/)")
    args = parser.parse_args()

    base_path = Path(args.data_dir).parent.as_posix()

    test_dataset = WaymoE2E(
        batch_size=args.batch_size,
        indexFile="index_train.pkl",
        data_dir=args.data_dir,
        images=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = "/scratch/gilbreth/shar1159/checkpoints/camera-e2e-epoch=02-val_loss=1.67.ckpt"

    in_dim = 16 * 6
    out_dim = 20 * 2
    model = NewModel(in_dim=in_dim, out_dim=out_dim)

    lit_model = LitModel.load_from_checkpoint(
        ckpt_path,
        model=model,
        lr=1e-3,  # not used for testing, but required by constructor
        loss_out_path="scene_loss.json",
    )

    lit_model = lit_model.cuda()
    lit_model.export_scene_loss_json(test_loader)


if __name__ == "__main__":
    main()