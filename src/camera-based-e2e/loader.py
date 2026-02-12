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
        images = True,
        n_items: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.images = images
        self.data_dir = data_dir
        self.seed = seed

        self.filename = ""
        self.file = None

        with open(indexFile, 'rb') as f:
            # NOTE: test does not have reference trajectories
            # We train on train and validate on val set
            self.indexes = pickle.load(f)

        # TODO: Determine how to sample specific subsets of the data that we care about.
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
        return len(self.indexes)
    
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

            self.file.seek(start_byte) # type: ignore
            protobuf = self.file.read(byte_length) # type: ignore
            frame.ParseFromString(protobuf)

            past = np.stack([frame.past_states.pos_x, frame.past_states.pos_y, frame.past_states.vel_x, frame.past_states.vel_y, frame.past_states.accel_x, frame.past_states.accel_y], axis=-1)
    

            future = np.stack([frame.future_states.pos_x, frame.future_states.pos_y], axis=-1)

            past = np.array(past, dtype=np.float32) # ensure consistent dtype
            future = np.array(future, dtype=np.float32)

            # For submission to waymo evaluation server
            name = frame.frame.context.name

            # Yield JPEG images as torch uint8 tensors so that PyTorch
            # DataLoader transfers them via shared memory instead of
            # pickle serialization — critical for multi-worker / DDP perf.
            jpeg_tensors = [
                torch.from_numpy(np.frombuffer(img.image, dtype=np.uint8).copy())
                for img in frame.frame.images
            ]

            yield {'PAST': past, 'FUTURE': future, 'IMAGES_JPEG': jpeg_tensors, 'INTENT': frame.intent, 'NAME': name}

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    # NOTE: Replace with your path
    DATA_DIR = '/anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0'
    BATCH_SIZE = 256
    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir = DATA_DIR, images=True)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        # collate_fn=collate_with_images, # from models.base_model
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
