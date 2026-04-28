import argparse
from datetime import datetime
import itertools
import math
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler

import torch
from pathlib import Path
import os
from torch.utils.data import BatchSampler

# Replace with your model defined in models/
from models.base_model import LitModel, collate_with_images
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures


class HomogeneousConcatBatchSampler(BatchSampler):
    """Emit batches from one ConcatDataset source at a time."""

    def __init__(
        self,
        dataset_lengths: tuple[int, int],
        batch_size: int,
        rank: int | None = None,
        world_size: int | None = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        source_ratio: tuple[int, int] = (1, 1),
        **kwargs,
    ):
        if len(dataset_lengths) != 2:
            raise ValueError(f"Expected exactly 2 datasets, got {len(dataset_lengths)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"Invalid rank/world_size pair: rank={rank}, world_size={world_size}"
            )

        self.lengths = dataset_lengths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.source_ratio = source_ratio

        self.offset0 = 0
        self.offset1 = dataset_lengths[0]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _num_samples_per_rank(self, length: int) -> int:
        if length <= 0:
            return 0
        if self.drop_last and length % self.world_size != 0:
            return math.ceil((length - self.world_size) / self.world_size)
        return math.ceil(length / self.world_size)

    def _make_rank_indices(self, start: int, length: int, rng: random.Random):
        idx = list(range(start, start + length))
        if self.shuffle:
            rng.shuffle(idx)
        num_samples = self._num_samples_per_rank(length)
        total_size = num_samples * self.world_size

        if self.drop_last:
            idx = idx[:total_size]
        elif total_size > len(idx):
            if not idx:
                return []
            padding_size = total_size - len(idx)
            repeats = math.ceil(padding_size / len(idx))
            idx += (idx * repeats)[:padding_size]

        return idx[self.rank : total_size : self.world_size]

    def _chunk(self, indices: list[int]) -> list[list[int]]:
        if self.drop_last:
            n_full = len(indices) // self.batch_size
            return [
                indices[i * self.batch_size : (i + 1) * self.batch_size]
                for i in range(n_full)
            ]

        out = []
        for i in range(0, len(indices), self.batch_size):
            out.append(indices[i : i + self.batch_size])
        return out

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        idx0 = self._make_rank_indices(self.offset0, self.lengths[0], rng)
        idx1 = self._make_rank_indices(self.offset1, self.lengths[1], rng)

        b0 = self._chunk(idx0)
        b1 = self._chunk(idx1)

        w0 = max(0, int(self.source_ratio[0]))
        w1 = max(0, int(self.source_ratio[1]))
        if w0 == 0 and w1 == 0:
            w0, w1 = 1, 1

        pattern = [0] * w0 + [1] * w1
        if not pattern:
            pattern = [0, 1]

        i0, i1 = 0, 0
        for source in itertools.cycle(pattern):
            if i0 >= len(b0) and i1 >= len(b1):
                break
            if source == 0:
                if i0 < len(b0):
                    yield b0[i0]
                    i0 += 1
            else:
                if i1 < len(b1):
                    yield b1[i1]
                    i1 += 1

    def __len__(self):
        def n_batches(length: int):
            per_rank = self._num_samples_per_rank(length)
            if self.drop_last:
                return per_rank // self.batch_size
            return math.ceil(per_rank / self.batch_size)

        return n_batches(self.lengths[0]) + n_batches(self.lengths[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="waymo",
        choices=["waymo", "nuscenes", "all"],
    )
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    # Data
    if args.dataset == "waymo":
        from loader import WaymoE2E

        train_dataset = WaymoE2E(
            indexFile="index_train.pkl", data_dir=args.data_dir, n_items=250_000
        )
        test_dataset = WaymoE2E(
            indexFile="index_val.pkl", data_dir=args.data_dir, n_items=25_000
        )
        nw = 0

    elif args.dataset == "nuscenes":
        from nuscenes_loader import NuScenesDataset

        train_dataset = NuScenesDataset(
            data_dir=args.data_dir, split="train", n_items=250_000
        )
        test_dataset = NuScenesDataset(
            data_dir=args.data_dir, split="val", n_items=25_000
        )
        nw = 16

    elif args.dataset == "all":
        from loader import WaymoE2E
        from nuscenes_loader import NuScenesDataset

        waymo_dir = os.getenv("WAYMO_DATA_DIR")
        nuscenes_dir = os.getenv("NUSCENES_DATA_DIR")

        waymo_train = WaymoE2E(
            indexFile="index_train.pkl", data_dir=waymo_dir, n_items=125_000
        )
        waymo_test = WaymoE2E(
            indexFile="index_val.pkl", data_dir=waymo_dir, n_items=12_500
        )

        nuscenes_train = NuScenesDataset(
            data_dir=nuscenes_dir, split="train", n_items=125_000
        )
        nuscenes_test = NuScenesDataset(
            data_dir=nuscenes_dir, split="val", n_items=12_500
        )

        train_dataset = torch.utils.data.ConcatDataset([waymo_train, nuscenes_train])
        test_dataset = torch.utils.data.ConcatDataset([waymo_test, nuscenes_test])
        nw = 16

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        train_batch_sampler = HomogeneousConcatBatchSampler(
            dataset_lengths=(len(waymo_train), len(nuscenes_train)),
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        )
        val_batch_sampler = HomogeneousConcatBatchSampler(
            dataset_lengths=(len(waymo_test), len(nuscenes_test)),
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=False,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.dataset == "all":
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=nw,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_with_images,
        )
        val_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=nw,
            batch_sampler=val_batch_sampler,
            collate_fn=collate_with_images,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=nw,
            collate_fn=collate_with_images,
        )
        val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=nw,
            collate_fn=collate_with_images,
        )

    # Model
    in_dim = 16 * 6
    out_dim = 20 * 2

    model = DeepMonocularModel(
        feature_extractor=SAMFeatures(
            model_name="timm/vit_pe_spatial_small_patch16_512.fb", frozen=True
        ),
        out_dim=out_dim,
        n_blocks=4,
    )

    if args.compile:
        model = torch.compile(model, mode="max-autotune")

    lit_model = LitModel(model=model, lr=args.lr)

    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    use_distributed_sampler = args.dataset != "all"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 16,
        profiler=SimpleProfiler(extended=True) if args.profile else None,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=Path(args.data_dir).parent.as_posix() + "/checkpoints",
                filename="camera-e2e-{epoch:02d}-{val_loss:.2f}",
            ),
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)
