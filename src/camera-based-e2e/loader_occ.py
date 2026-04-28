import torch
from torch.utils.data import Dataset
from protos import e2e_pb2
import pickle
import os
import numpy as np
from typing import Optional
import random

devices = ['cuda:0', 'cuda:1']

random.seed(42) # Deterministic

class WaymoE2E(Dataset):
    def __init__(
        self,
        indexFile = 'index.pkl',
        data_dir='./dataset',
        n_items: Optional[int] = None,
        seed: Optional[int] = None,
        occ_root: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.seed = seed

        self.filename = ""
        self.file = None

        with open(indexFile, 'rb') as f:
            # NOTE: test does not have reference trajectories
            # We train on train and validate on val set
            raw_indexes = pickle.load(f)

        self.indexes = [(orig_idx, item) for orig_idx, item in enumerate(raw_indexes)]

        # TODO: Determine how to sample specific subsets of the data that we care about.
        if n_items is not None and n_items < len(self.indexes):
            self.indexes = self.indexes[:n_items]

        self.occ_root = occ_root
        self.occ_index = None
        self.split = "train" if "train" in os.path.basename(indexFile) else "val" if "val" in os.path.basename(indexFile) else "test"

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        frame = e2e_pb2.E2EDFrame()  # type: ignore
        orig_idx, (filename, start_byte, byte_length) = self.indexes[idx]

        if self.filename != filename:
            if self.file:
                self.file.close()
                del self.file
            self.file = open(os.path.join(self.data_dir, filename), 'rb')
            self.filename = filename

        self.file.seek(start_byte) # type: ignore
        protobuf = self.file.read(byte_length) # type: ignore
        frame.ParseFromString(protobuf)

        past = np.stack(
            [
                frame.past_states.pos_x,
                frame.past_states.pos_y,
                frame.past_states.vel_x,
                frame.past_states.vel_y,
                frame.past_states.accel_x,
                frame.past_states.accel_y,
            ],
            axis=-1,
        )

        future = np.stack([frame.future_states.pos_x, frame.future_states.pos_y], axis=-1)

        past = np.array(past, dtype=np.float32) # ensure consistent dtype
        future = np.array(future, dtype=np.float32)

        # For submission to waymo evaluation server
        name = frame.frame.context.name

        # Return JPEG images as torch uint8 tensors so DataLoader can use shared memory.
        jpeg_tensors = [
            torch.from_numpy(np.frombuffer(img.image, dtype=np.uint8).copy())
            for img in frame.frame.images
        ]

        occ = None
        if self.occ_root is not None:
            occ_path = os.path.join(self.occ_root, self.split, f"occ_{orig_idx:07d}.npy")
            if not os.path.exists(occ_path):
                raise FileNotFoundError(f"Missing OCC file for index {orig_idx}: {occ_path}")
            occ = np.load(occ_path).astype(np.uint8)

        return {'PAST': past, 'FUTURE': future, 'IMAGES_JPEG': jpeg_tensors, 'INTENT': frame.intent, 'NAME': name, 'OCC': occ}


def collate_with_images(batch):
    """Collate that keeps IMAGES_JPEG as a list-of-lists (variable-size JPEG
    bytes cannot be stacked) and delegates everything else to default_collate
    when OCC is not present."""
    from torch.utils.data.dataloader import default_collate

    if batch[0].get("OCC", None) is None:
        images = [sample.pop('IMAGES_JPEG') for sample in batch]
        for sample in batch:
            sample.pop('OCC', None)
        collated = default_collate(batch)
        collated['IMAGES_JPEG'] = images  # list[list[Tensor]], one inner list per sample
        return collated

    past = [torch.as_tensor(b["PAST"], dtype=torch.float32) for b in batch]
    future = [torch.as_tensor(b["FUTURE"], dtype=torch.float32) for b in batch]
    intent = torch.as_tensor([b["INTENT"] for b in batch])
    names = [b["NAME"] for b in batch]

    cams = list(zip(*[b["IMAGES_JPEG"] for b in batch]))  # per-camera tuples
    images_jpeg = [list(cam_imgs) for cam_imgs in cams]  # stay on CPU

    out = {
        "PAST": torch.stack(past, dim=0),
        "FUTURE": torch.stack(future, dim=0),
        "INTENT": intent,
        "IMAGES_JPEG": images_jpeg,
        "NAME": names,
        "OCC": torch.stack(
            [torch.as_tensor(b["OCC"], dtype=torch.long) for b in batch],
            dim=0,
        ),
    }

    return out


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    # NOTE: Replace with your path
    DATA_DIR = '/anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0'
    BATCH_SIZE = 256
    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir = DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=collate_with_images,
        pin_memory=True, # causes error
    )
    # next(iter(loader))

    def main():
        # start = time.time()
        for batch_of_frames in tqdm(loader):
            # print(batch_of_frames["INTENT"])
            # print(batch_of_frames.keys(), [b.shape for b in batch_of_frames.values() if isinstance(b, torch.Tensor)])
            pass
        # print("Total Time:", time.time()-start)

    import cProfile
    main()
