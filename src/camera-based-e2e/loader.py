import torch
from torch.utils.data import IterableDataset
from protos import e2e_pb2
import torchvision
from sklearn.cluster import MiniBatchKMeans
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

            yield {'PAST': past, 'FUTURE': future, 'IMAGES': [self.decode_img(images.image) for images in frame.frame.images], 'INTENT': frame.intent, 'NAME': name}

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    # NOTE: Replace with your path
    DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/'
    BATCH_SIZE = 64
    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir = DATA_DIR, images=False)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    def main():
        all_pasts = []
        all_futures = []
        
        min_p, max_p = None, None
        min_f, max_f = None, None

        print(f"Processing {len(loader.dataset)} samples...")

        for batch in tqdm(loader):
            past = batch["PAST"]    # [B, 10, 6] or similar
            future = batch["FUTURE"] # [B, 20, 2]

            b_min_p, b_max_p = torch.amin(past, dim=0), torch.amax(past, dim=0)
            b_min_f, b_max_f = torch.amin(future, dim=0), torch.amax(future, dim=0)

            if min_p is None:
                min_p, max_p = b_min_p, b_max_p
                min_f, max_f = b_min_f, b_max_f
            else:
                min_p, max_p = torch.min(min_p, b_min_p), torch.max(max_p, b_max_p)
                min_f, max_f = torch.min(min_f, b_min_f), torch.max(max_f, b_max_f)

            all_pasts.append(past.cpu().numpy().astype(np.float32))
            all_futures.append(future.cpu().numpy().astype(np.float32))

        print("Concatenating and saving full datasets...")
        full_pasts = np.concatenate(all_pasts, axis=0)
        full_futures = np.concatenate(all_futures, axis=0)

        np.save('all_past_states.npy', full_pasts)
        np.save('all_future_states.npy', full_futures)
        
        torch.save({'min': min_p, 'max': max_p}, 'past_min_max.pt')
        torch.save({'min': min_f, 'max': max_f}, 'future_min_max.pt')

        print("Starting MiniBatchKMeans...")
        n_samples = full_futures.shape[0]
        flattened_futures = full_futures.reshape(n_samples, -1)

        n_clusters = 20
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init='auto', random_state=42)
        kmeans.fit(flattened_futures)

        centroids = kmeans.cluster_centers_.reshape(n_clusters, 20, 2)
        torch.save(torch.from_numpy(centroids).float(), 'future_clusters.pt')
        
        print("Done! Files saved: all_past_states.npy, all_future_states.npy, past_min_max.pt, future_clusters.pt")
    import cProfile
    main()
