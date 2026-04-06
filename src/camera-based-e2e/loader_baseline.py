import torch
from torch.utils.data import Dataset
from protos import e2e_pb2
import pickle
import os
import numpy as np
from typing import Optional
import random

devices = ['cuda:0', 'cuda:1']

random.seed(42)  # Deterministic


class WaymoE2E(Dataset):
    def __init__(
        self,
        indexFile='index.pkl',
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

        # Use the first n_items in dataset order
        if n_items is not None and n_items < len(self.indexes):
            self.indexes = self.indexes[:n_items]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        frame = e2e_pb2.E2EDFrame()  # type: ignore
        filename, start_byte, byte_length = self.indexes[idx]

        if self.filename != filename:
            if self.file:
                self.file.close()
                del self.file
            self.file = open(os.path.join(self.data_dir, filename), 'rb')
            self.filename = filename

        self.file.seek(start_byte)  # type: ignore
        protobuf = self.file.read(byte_length)  # type: ignore
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

        future = np.stack(
            [frame.future_states.pos_x, frame.future_states.pos_y], axis=-1
        )

        past = np.array(past, dtype=np.float32)
        future = np.array(future, dtype=np.float32)

        # For submission to waymo evaluation server
        name = frame.frame.context.name

        # Return JPEG images as torch uint8 tensors so DataLoader can use shared memory.
        jpeg_tensors = [
            torch.from_numpy(np.frombuffer(img.image, dtype=np.uint8).copy())
            for img in frame.frame.images
        ]

        return {
            'PAST': past,
            'FUTURE': future,
            'IMAGES_JPEG': jpeg_tensors,
            'INTENT': frame.intent,
            'NAME': name,
        }


def collate_with_images(batch):
    """Collate that keeps IMAGES_JPEG as a list-of-lists (variable-size JPEG
    bytes cannot be stacked) and delegates everything else to default_collate."""
    from torch.utils.data.dataloader import default_collate

    images = [sample.pop('IMAGES_JPEG') for sample in batch]
    collated = default_collate(batch)
    collated['IMAGES_JPEG'] = images  # list[list[Tensor]], one inner list per sample
    return collated


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # NOTE: Replace with your path
    DATA_DIR = '/anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0'
    BATCH_SIZE = 256
    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir=DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=collate_with_images,
        pin_memory=True,  # causes error
    )

    def main():
        for batch_of_frames in tqdm(loader):
            pass

    import cProfile
    main()
