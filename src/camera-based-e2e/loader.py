import torch
from torch.utils.data import IterableDataset
from protos import e2e_pb2
import pickle
import os
import numpy as np
from typing import Optional
import random

devices = ['cuda:0', 'cuda:1']

random.seed(42) # Deterministic

class WaymoE2E(IterableDataset):
    def __init__(
        self,
        indexFile = 'index.pkl',
        data_dir='./dataset',
        n_items: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.seed = seed

        self.filename = ""
        self.file = None

        with open(indexFile, 'rb') as f:
            # NOTE: test does not have reference trajectories
            # We train on train and validate on val set
            self.indexes = pickle.load(f)

        if n_items is not None and n_items < len(self.indexes):
            total = len(self.indexes)
            # pick a deterministic contiguous block when a seed is provided
            rng = random.Random(seed) if seed is not None else random
            start = rng.randint(0, total - n_items)
            self.indexes = self.indexes[start : start + n_items]

    def global_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

    def __len__(self):
        _, world_size = self.global_rank()
        return len(self.indexes) // world_size

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            start, step = self.global_rank()
        else:
            rank, world_size = self.global_rank()
            global_worker_id = rank * worker.num_workers + worker.id
            global_num_workers = world_size * worker.num_workers
            start, step = global_worker_id, global_num_workers

        for idx in range(start, len(self.indexes), step):
            frame = e2e_pb2.E2EDFrame()  # type: ignore
            filename, start_byte, byte_length = self.indexes[idx]

            if self.filename != filename:
                if self.file:
                    self.file.close()
                    del self.file
                self.file = open(os.path.join(self.data_dir, filename), 'rb')
                self.filename = filename

            self.file.seek(start_byte)   # type: ignore
            protobuf = self.file.read(byte_length)  # type: ignore
            frame.ParseFromString(protobuf)

            past = np.stack([
                frame.past_states.pos_x,
                frame.past_states.pos_y,
                frame.past_states.vel_x,
                frame.past_states.vel_y,
                frame.past_states.accel_x,
                frame.past_states.accel_y,
            ], axis=-1)

            future = np.stack([
                frame.future_states.pos_x,
                frame.future_states.pos_y,
            ], axis=-1)

            past   = np.array(past,   dtype=np.float32)
            future = np.array(future, dtype=np.float32)

            # For submission to waymo evaluation server
            name = frame.frame.context.name

            # Yield JPEG images as raw uint8 byte tensors so that PyTorch
            # DataLoader transfers them via shared memory instead of pickle
            # serialisation — critical for multi-worker / DDP performance.
            jpeg_tensors = [
                torch.from_numpy(np.frombuffer(img.image, dtype=np.uint8).copy())
                for img in frame.frame.images
            ]

            yield {
                'PAST':        past,
                'FUTURE':      future,
                'IMAGES_JPEG': jpeg_tensors,   # list[uint8 tensor], one per camera
                'INTENT':      frame.intent,
                'NAME':        name,
                'FRAME':       frame.frame,    # Waymo Frame proto (intrinsics/extrinsics)
            }


# ---------------------------------------------------------------------------
# collate_fn must be defined at module level (outside __main__) so it can
# be imported by generate_pcd.py, train.py, submission.py, etc.
# ---------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate for WaymoE2E batches.
    Handles variable-length JPEG byte tensors and non-collatable Frame protos.

    Args:
        batch: list of dicts from WaymoE2E.__iter__

    Returns:
        dict with keys: PAST, FUTURE, IMAGES_JPEG, INTENT, NAME, FRAME
        - PAST, FUTURE : stacked float32 tensors  [B, ...]
        - IMAGES_JPEG  : list[list[uint8 tensor]] — [num_cameras][B]
        - INTENT       : list of length B
        - NAME         : list of length B
        - FRAME        : list of Frame protos, length B  (not stackable)
    """
    return {
        'PAST':   torch.from_numpy(np.stack([b['PAST']   for b in batch])),
        'FUTURE': torch.from_numpy(np.stack([b['FUTURE'] for b in batch])),
        'IMAGES_JPEG': [
            [b['IMAGES_JPEG'][cam_idx] for b in batch]
            for cam_idx in range(len(batch[0]['IMAGES_JPEG']))
        ],
        'INTENT': [b['INTENT'] for b in batch],
        'NAME':   [b['NAME']   for b in batch],
        'FRAME':  [b['FRAME']  for b in batch],
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    DATA_DIR = 'scratch/gilbreth/kumar753/robotvision/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0/'
    BATCH_SIZE = 256

    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir=DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,  # pin_memory=True errors with proto objects in batch
    )

    def main():
        for batch in tqdm(loader):
            pass

    main()
