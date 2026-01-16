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
from models.base_model import collate_with_images

devices = ['cuda:0', 'cuda:1']

random.seed(42) # Deterministic

class WaymoE2E(IterableDataset): 
    def __init__(
        self,
        indexFile = 'preference_cache.pkl',
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

        if n_items is not None and n_items < len(self.indexes):
            total = len(self.indexes)
            # pick a deterministic contiguous block when a seed is provided
            rng = random.Random(seed) if seed is not None else random
            start = rng.randint(0, total - n_items)
            self.indexes = self.indexes[start : start + n_items]

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
        # spawns a new worker, assigns worker.id and num_workers
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            start, step = 0, 1
        else:
            start, step = worker.id, worker.num_workers

        # this code makes each worker read a subset of teh data (worker 1 = 0, 16, 32..., worker 2 = 1, 17, 33...)
        for idx in range(start, len(self.indexes), step):
            frame = e2e_pb2.E2EDFrame() 
            filename, start_byte, byte_length = self.indexes[idx]

            if self.filename != filename:
                if self.file:
                    self.file.close()
                    del self.file
                self.file = open(os.path.join(self.data_dir, filename), 'rb')
                self.filename = filename

            self.file.seek(start_byte) 
            protobuf = self.file.read(byte_length) 
            frame.ParseFromString(protobuf)

            past = np.stack([frame.past_states.pos_x, frame.past_states.pos_y, frame.past_states.vel_x, frame.past_states.vel_y, frame.past_states.accel_x, frame.past_states.accel_y], axis=-1)
    
            future = np.stack([frame.future_states.pos_x, frame.future_states.pos_y], axis=-1)

            past = np.array(past, dtype=np.float32) # ensure consistent dtype
            future = np.array(future, dtype=np.float32)

            # For submission to waymo evaluation server
            name = frame.frame.context.name

            for traj in frame.preference_trajectories:
                # stack x, y
                traj_array = np.stack([traj.pos_x, traj.pos_y], axis=-1)  # shape (T, 2), T = timesteps
                yield {'PAST': past, 'FUTURE': future, 'IMAGES': [self.decode_img(images.image) for images in frame.frame.images], 'INTENT': frame.intent, 'NAME': name, 'PREF_TRAJ': traj_array, 'PREF_SCORE': traj.preference_score}

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    # NOTE: Replace with your path
    DATA_DIR = '/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0/'
    BATCH_SIZE = 32
    dataset = WaymoE2E(indexFile="preference_cache.pkl", data_dir = DATA_DIR, images=True)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=16, 
        collate_fn=collate_with_images
    )
    
    #now we have this laoder class that I can iterate over like an array
    #it will return dictionaries with keys PAST, FUTURE, IMAGES, INTENT

    def main():
        # start = time.time()
        for batch_of_frames in tqdm(loader):
            # print(batch_of_frames["INTENT"])
            # print(batch_of_frames.keys(), [b.shape for b in batch_of_frames.values() if isinstance(b, torch.Tensor)])
            pass
        # print("Total Time:", time.time()-start)
        print(len(loader))
    
    import cProfile
    main()