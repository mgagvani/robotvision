import torch
from torch.utils.data import IterableDataset
from protos import e2e_pb2
import torchvision
import pickle
import struct
import os
import numpy as np
from PIL import Image
from io import BytesIO 
import cv2 
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
        
        # Precomputed shard loading options
        use_precomputed: bool = False,
        precomputed_dir: Optional[str] = None,
        shard_size: int = 1000,
    ):
        self.images = images
        self.data_dir = data_dir
        self.seed = seed
        
        # Precomputed options
        self.use_precomputed = use_precomputed
        self.precomputed_dir = precomputed_dir
        self.shard_size = shard_size
        
        # Track current shard to enable cleanup after moving to next shard
        self._current_shard_idx = None
        self._current_shard_data = None
        
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

        # Build index mapping for precomputed data
        if self.use_precomputed and self.precomputed_dir:
            self._build_precomputed_index()

    def __del__(self):
        """Ensure file handles are closed on object destruction."""
        if self.file:
            self.file.close()
            self.file = None

    def _build_precomputed_index(self):
        shard_files = sorted([f for f in os.listdir(self.precomputed_dir) if f.startswith('shard_') and f.endswith('.pt')])
        
        self._shard_files = shard_files
        self._num_shards = len(shard_files)
        
        print(f"Precomputed: {len(self.indexes)} samples across {self._num_shards} shards (shard_size={self.shard_size})")

    def _cleanup_previous_shard(self, new_shard_idx: int):
        """Clear reference to previous shard to free memory when moving to next shard."""
        if self._current_shard_idx is not None and self._current_shard_idx != new_shard_idx:
            self._current_shard_data = None  # Allow GC to collect previous shard

    def _load_shard(self, shard_idx: int, shard_path: str):
        """Load a shard from disk. No caching needed since each shard is used only once."""
        return torch.load(shard_path, map_location='cpu')

    def _get_precomputed_data(self, idx: int):
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        
        if shard_idx >= self._num_shards:
            return {'depth': None, 'obj_det': None, 'lane_det': None, 'tokens': None}
        
        # Clean up previous shard before loading new one
        self._cleanup_previous_shard(shard_idx)
        
        # Load new shard if needed
        if self._current_shard_idx != shard_idx:
            shard_path = os.path.join(self.precomputed_dir, self._shard_files[shard_idx])
            self._current_shard_data = self._load_shard(shard_idx, shard_path)
            self._current_shard_idx = shard_idx
        
        # Direct indexing into the list
        if local_idx >= len(self._current_shard_data):
            return {'depth': None, 'obj_det': None, 'lane_det': None, 'tokens': None}
        
        sample = self._current_shard_data[local_idx]
        
        return {
            'depth': sample['depth'],
            'obj_det': sample['obj_det'],
            'lane_det': sample['lane_det'],
            'tokens': sample['tokens']
        }

    def decode_img(self, img):
        if not self.images:
            return np.array([])
        
        img_tensor = torch.from_numpy(np.frombuffer(img, dtype=np.uint8).copy()) 
        gpu_tensors_list = torchvision.io.decode_jpeg(
            img_tensor, 
            mode=torchvision.io.ImageReadMode.UNCHANGED,
            device= 'cpu' #['cuda:0', 'cuda:1'][torch.utils.data.get_worker_info().id%2]
        )
        # img_array = np.frombuffer(img, np.uint8)
        return gpu_tensors_list
    
    def __len__(self):
        return len(self.indexes)
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            start, step = 0, 1
        else:
            start, step = worker.id, worker.num_workers

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

            # Build base batch
            batch = {'PAST': past, 'FUTURE': future, 'IMAGES': [self.decode_img(images.image) for images in frame.frame.images], 'INTENT': frame.intent, 'NAME': name}
            
            # Add precomputed perception data if enabled
            if self.use_precomputed and self.precomputed_dir:
                precomputed_data = self._get_precomputed_data(idx)
                batch['TOKENS'] = precomputed_data['tokens']
                batch['PRECOMPUTED_DEPTH'] = precomputed_data['depth']
                batch['PRECOMPUTED_OBJ_DET'] = precomputed_data['obj_det']
                batch['PRECOMPUTED_LANE'] = precomputed_data['lane_det']
            
            yield batch

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    # NOTE: Replace with your path
    DATA_DIR = '/scratch/gilbreth/mgagvani/wod/waymo_open_dataset_end_to_end_camera_v_1_0_0/'
    BATCH_SIZE = 32
    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir = DATA_DIR, images=True)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=16,
    )
    
    def main():
        # start = time.time()
        for batch_of_frames in tqdm(loader):
            # print(batch_of_frames["INTENT"])
            # print(batch_of_frames.keys(), [b.shape for b in batch_of_frames.values() if isinstance(b, torch.Tensor)])
            pass
        # print("Total Time:", time.time()-start)
    
    import cProfile
    main()
