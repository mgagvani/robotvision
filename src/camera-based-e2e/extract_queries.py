import argparse
import torch
from models.base_model import LitModel, collate_with_images
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures
from loader import WaymoE2E

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=30000, help="runs of deep monocular model")
parser.add_argument("--ckpt", type=str, 
                    default="/scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e/checkpoints/camera-e2e-epoch=04-val_loss=2.90.ckpt", 
                    help="checkpoint of deep monocular")
parser.add_argument("--batch_size", type=int, default=16, help="number of batches in testing")
parser.add_argument("--data_dir", type=str, 
                    default="/scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0", 
                    help="Waymo data directory")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_dim = 16 * 6  # Past: (B, 16, 6)
out_dim = 20 * 2  # Future: (B, 20, 2)
ckpt_path = args.ckpt

model = DeepMonocularModel(
    feature_extractor=SAMFeatures(
        model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
    ),
    out_dim=out_dim,
    n_blocks=4,
)

lit_model = LitModel.load_from_checkpoint(
    ckpt_path,
    model=model,
    lr=1e-4,   # should match training setup
)

lit_model.eval()
lit_model.freeze()
lit_model.to(device)

net = lit_model.model
net.to(device)
net.eval()

test_dataset = WaymoE2E(
            indexFile="index_val.pkl", data_dir=args.data_dir, n_items=args.runs
        )

test_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=0,
            batch_size=args.batch_size,
            collate_fn=collate_with_images,
            persistent_workers=False,
            pin_memory=False,
        )

queries = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        batch["PAST"] = batch["PAST"].to(device)
        batch["INTENT"] = batch["INTENT"].to(device)
        batch["FUTURE"] = batch["FUTURE"].to(device)
        batch["IMAGES"] = lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)

        preds = net(batch)

        print(f"batch {i}")
        print("query shape:", preds["query"].shape)

        queries.append(preds["query"].detach().cpu())

torch.save(queries, "query_data.pt")