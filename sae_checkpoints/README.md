# SAE Checkpoints

Copied from:
`/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1/model`

Files:
- `sae_block_0.pt` -> `planner_query_tok_block_0`
- `sae_block_1.pt` -> `planner_query_tok_block_1`
- `sae_block_2.pt` -> `planner_query_tok_block_2`
- `sae_block_3.pt` -> `planner_query_tok_block_3`

Minimal end-to-end example:

```python
from pathlib import Path
import sys

import torch

REPO_ROOT = Path("/home/mgagvani/robotvision")
sys.path.insert(0, str(REPO_ROOT / "src/camera-based-e2e"))

from extract_planner_tok import load_model
from loader import WaymoE2E
from models.base_model import collate_with_images
from sae_utils import build_sae_from_checkpoint

device = torch.device("cuda")
sae_block = 2

planner_model, lit_model = load_model(
    str(REPO_ROOT / "src/camera-based-e2e/camera-e2e-epoch=04-val_loss=2.90.ckpt"),
    device,
)

sae_ckpt = torch.load(
    REPO_ROOT / "sae_checkpoints" / f"sae_block_{sae_block}.pt",
    map_location="cpu",
)
sae = build_sae_from_checkpoint(sae_ckpt).to(device).eval()

dataset = WaymoE2E(
    indexFile="index_val.pkl",
    data_dir="/path/to/waymo_open_dataset_end_to_end_camera_v_1_0_0",
    n_items=1,
)
batch = collate_with_images([dataset[0]])

model_inputs = {
    "PAST": batch["PAST"].to(device),
    "IMAGES": lit_model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device),
    "INTENT": batch["INTENT"].to(device),
}

with torch.no_grad():
    block_queries, tokens, _ = planner_model.collect_block_query_tokens(
        model_inputs["PAST"],
        model_inputs["IMAGES"],
        model_inputs["INTENT"],
    )
    q = block_queries[sae_block].squeeze(1)
    z = sae.encode(q)
    q_hat = sae.decode_to_input(z, reference_x=q)

    out = planner_model.forward_from_block_query_tok(
        q_hat,
        model_inputs["PAST"],
        tokens,
        start_block=sae_block,
    )

print(sae_ckpt["token_key"], out["trajectory"].shape, out["scores"].shape)
```
