"""
Train VLMTrajectoryModel (vision encoder -> adapter -> decoder -> output head) on Waymo E2E.
Run from src/camera-based-e2e: python vlm_new/train.py --data_dir /path/to/data
"""

import argparse
import os
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Run from src/camera-based-e2e so loader and models resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import WaymoE2E
from models.base_model import LitModel, collate_with_images

from vlm_new.model import VLMTrajectoryModel
from vlm_new.vision_encoder import VisionEncoder, build_timm_patch_encoder
from vlm_new.adapter import Adapter, CONTEXT_INPUT_DIM
from vlm_new.decoder import QwenDecoder
from vlm_new.output_head import TrajectoryHead, NUM_TRAJECTORY_QUERIES

def build_model(args) -> VLMTrajectoryModel:
    llm_dim = getattr(args, "llm_dim", 256)
    model_name = getattr(args, "backbone", "vit_tiny_patch16_224")

    backbone = build_timm_patch_encoder(
        model_name=model_name,
        pretrained=True,
        global_pool="",
        num_classes=0,
    )
    if getattr(backbone, "default_cfg", None):
        inp = backbone.default_cfg.get("input_size", (3, 224, 224))
        patch_size = inp[1]
    else:
        patch_size = getattr(args, "patch_size", 224)

    vision_encoder = VisionEncoder(
        backbone,
        patch_size=patch_size,
        num_patches=2,
        frozen=True,
    )
    vision_dim = vision_encoder.embed_dim

    adapter = Adapter(
        vision_dim=vision_dim,
        llm_embed_dim=llm_dim,
        num_context_tokens=4,
        num_vision_tokens_after_downsample=getattr(args, "num_vision_tokens", None),
        context_input_dim=CONTEXT_INPUT_DIM,
    )

    use_lora = getattr(args, "use_lora", False)
    train_decoder = getattr(args, "train_decoder", False)
    freeze_decoder = not use_lora and not train_decoder
    decoder = QwenDecoder(
        model_name_or_path=getattr(args, "decoder_model", "Qwen/Qwen3-0.6B"),
        embed_dim=llm_dim,
        freeze=freeze_decoder,
        use_lora=use_lora,
        lora_r=getattr(args, "lora_r", 64),
        lora_alpha=getattr(args, "lora_alpha", 16),
    )

    output_head = TrajectoryHead(embed_dim=llm_dim, num_queries=NUM_TRAJECTORY_QUERIES)

    return VLMTrajectoryModel(
        vision_encoder=vision_encoder,
        adapter=adapter,
        decoder=decoder,
        output_head=output_head,
        num_trajectory_queries=NUM_TRAJECTORY_QUERIES,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/scratch/gilbreth/$USER/waymo-data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--decoder_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_decoder", action="store_true", help="Full fine-tune decoder (default: frozen)")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA on decoder instead of freezing")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--llm_dim", type=int, default=256)
    parser.add_argument("--num_vision_tokens", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="logs/vlm_new")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    data_dir = os.path.expandvars(args.data_dir)
    train_dataset = WaymoE2E(
        indexFile="index_train.pkl",
        data_dir=data_dir,
        images=True,
        n_items=args.limit_train,
        seed=args.seed,
    )
    val_dataset = WaymoE2E(
        indexFile="index_val.pkl",
        data_dir=data_dir,
        images=True,
        n_items=args.limit_val,
        seed=args.seed + 1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=collate_with_images,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_with_images,
    )

    model = build_model(args)
    lit = LitModel(model=model, lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = CSVLogger(save_dir=args.save_dir, name="")
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "checkpoints"),
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(lit, train_loader, val_loader)


if __name__ == "__main__":
    main()
