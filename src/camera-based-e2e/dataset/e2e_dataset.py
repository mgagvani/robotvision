import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm


DEFAULT_CAMERA_ORDER = [
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR_RIGHT",
    "REAR",
    "REAR_LEFT",
]


class E2EDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        images: bool = True,
        n_items: Optional[int] = None,
        seed: Optional[int] = None,
        image_order: Optional[List[str]] = None,
        return_context_name: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.images = images
        self.image_order = image_order or DEFAULT_CAMERA_ORDER
        self.return_context_name = return_context_name

        self.split_dir = self.root_dir / split
        manifest_path = self.split_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.entries: List[Dict[str, object]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(json.loads(line))

        if n_items is not None and n_items < len(self.entries):
            total = len(self.entries)
            rng = random.Random(seed) if seed is not None else random
            start = rng.randint(0, total - n_items)
            self.entries = self.entries[start : start + n_items]

    def __len__(self) -> int:
        return len(self.entries)

    def _decode_image(self, path: Path) -> torch.Tensor:
        img_bytes = path.read_bytes()
        img_tensor = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.uint8).copy())
        return torchvision.io.decode_jpeg(
            img_tensor, mode=torchvision.io.ImageReadMode.UNCHANGED
        )

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        state_path = self.split_dir / entry["state"]
        with np.load(state_path) as data:
            past = data["past"].astype(np.float32)
            future = data["future"].astype(np.float32)
            intent = int(data["intent"])

        images: List[object] = []
        images_map = entry.get("images", {})
        if self.images:
            for cam in self.image_order:
                rel = images_map.get(cam)
                if rel is None:
                    images.append(np.array([]))
                    continue
                images.append(self._decode_image(self.split_dir / rel))
        else:
            images = [np.array([]) for _ in self.image_order]

        name = entry.get("id")
        if self.return_context_name:
            meta_path = self.split_dir / entry["meta"]
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            name = meta.get("context_name", name)

        return {
            "PAST": past,
            "FUTURE": future,
            "IMAGES": images,
            "INTENT": intent,
            "NAME": name,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # NOTE: Replace with your path
    DATA_DIR = "/scratch/gilbreth/$USER/wod/e2e_exported"
    BATCH_SIZE = 32
    dataset = E2EDataset(root_dir=DATA_DIR, split="train", images=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    def main() -> None:
        for _ in tqdm(loader):
            pass

    main()
