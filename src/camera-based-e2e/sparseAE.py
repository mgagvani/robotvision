import argparse

import pytorch_lightning as pl
from models.base_model import LitModel, collate_with_images
import torch
import torch.nn.functional as F

from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures

from loader import WaymoE2E

class SparseAE(pl.LightningModule):
    def __init__(self, target_model, input_dim, dict_size, l1_coeff=1e-3, compile_sae=False):
        super().__init__()
        self.target_model = target_model
        
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.target_model.eval()
        
        # SAE: expansion -> ReLU -> contraction
        self.encoder = torch.nn.Linear(input_dim, dict_size)
        self.decoder = torch.nn.Linear(dict_size, input_dim)
        self.l1_coeff = l1_coeff
        self.compile_sae = compile_sae
        
        self.internal_acts = None

        if self.compile_sae and hasattr(torch, "compile"):
            # Compile only the trainable SAE path; keep the hooked target model untouched.
            self.encoder = torch.compile(self.encoder, mode="max-autotune")
            self.decoder = torch.compile(self.decoder, mode="max-autotune")

    def hook_fn(self, module, input, output):
        # Store activations from the base model
        self.internal_acts = output.detach()

    def _camera_major_to_batch_major(self, images_jpeg):
        if not isinstance(images_jpeg, (list, tuple)):
            return images_jpeg
        if len(images_jpeg) == 0:
            return images_jpeg
        if isinstance(images_jpeg[0], torch.Tensor):
            return [list(images_jpeg)]

        # base_model.collate_with_images returns camera-major:
        # outer=list of cameras, inner=list over batch.
        first = images_jpeg[0]
        if not isinstance(first, (list, tuple)):
            return images_jpeg

        batch_size = len(first)
        if batch_size == 0:
            return [[] for _ in range(0)]

        return [
            [images_jpeg[cam_idx][sample_idx] for cam_idx in range(len(images_jpeg))]
            for sample_idx in range(batch_size)
        ]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not isinstance(batch, dict):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        if "IMAGES_JPEG" in batch:
            images_jpeg = self._camera_major_to_batch_major(batch["IMAGES_JPEG"])
            batch_wo_jpeg = dict(batch)
            batch_wo_jpeg.pop("IMAGES_JPEG", None)
            moved = super().transfer_batch_to_device(batch_wo_jpeg, device, dataloader_idx)
            moved["IMAGES"] = self.target_model.decode_batch_jpeg(images_jpeg, device=device)
            return moved

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(self, batch, batch_idx):
        past, intent = batch['PAST'], batch['INTENT']
        if not isinstance(past, torch.Tensor):
            past = torch.as_tensor(past, dtype=torch.float32)
        if not isinstance(intent, torch.Tensor):
            intent = torch.as_tensor(intent)
        device = self.device if hasattr(self, 'device') else None
        past = past.to(device=device)
        intent = intent.to(device=device)
        self.target_model.eval()
        self.internal_acts = None
        if "IMAGES" in batch:
            images = [img.to(device=device) for img in batch["IMAGES"]]
        else:
            images_jpeg = self._camera_major_to_batch_major(batch["IMAGES_JPEG"])
            images = self.target_model.decode_batch_jpeg(images_jpeg, device=device)
            
        model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}

        with torch.no_grad():
            _ = self.target_model(model_inputs)
        
        if self.internal_acts is None:
            raise RuntimeError("No activations captured from target model hook; ensure the hook is registered and model forward ran.")
        x = self.internal_acts
        
        x_centered = x - self.decoder.bias
        
        hidden = torch.relu(self.encoder(x_centered))
        
        reconstructed = self.decoder(hidden)
        
        mse_loss = F.mse_loss(reconstructed, x)
        l1_loss = hidden.abs().sum(dim=-1).mean()
        total_loss = mse_loss + (self.l1_coeff * l1_loss)
        
        self.log("sae/total_loss", total_loss, prog_bar=True)
        self.log("sae/mse_loss", mse_loss)
        self.log("sae/l1_loss", l1_loss)
        
        l0 = (hidden > 0).float().sum(dim=-1).mean()
        self.log("sae/l0_sparsity", l0, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        # Only optimize SAE parameters; target_model is frozen and should not be in the optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        return torch.optim.Adam(params, lr=1e-4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the SAE encoder/decoder with torch.compile")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of SAE epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for SAE training")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./pretrained/camera-e2e-epoch=04-val_loss=2.90.ckpt",
        help="Checkpoint for the frozen target model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/",
        help="Waymo dataset directory",
    )
    args = parser.parse_args()

    submodel = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=40,
        n_blocks=4,
    )
    model = LitModel.load_from_checkpoint(args.checkpoint_path, model = submodel)
    # for name, layer in model.model.named_modules():
    #     print(f"Name: {name} | Type: {type(layer)}")

    target_layer = model.model.blocks[3].mlp[2]
    print(target_layer)
    sae = SparseAE(
        target_model=model,
        input_dim=target_layer.out_features,
        dict_size=4096,
        compile_sae=args.compile,
    )
    target_layer.register_forward_hook(sae.hook_fn)

    waymo_train = WaymoE2E(
            indexFile="index_train.pkl", data_dir=args.data_dir, n_items=80_000
        )
    from torch.utils.data import DataLoader

    # Wrap dataset in a DataLoader (trainer.fit expects a DataLoader or LightningDataModule)
    train_loader = DataLoader(
        waymo_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        pin_memory=False,
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs)

    trainer.fit(sae, train_dataloaders=train_loader)
