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

class WaymoE2E(IterableDataset): 
    def __init__(self, batch_size, indexFile = 'index.pkl', data_dir='./dataset', images = True, n_items: Optional[int] = None):
        self.images = images
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.filename = ""
        self.file = None

        with open(indexFile, 'rb') as f:
            self.indexes = pickle.load(f)

        # TODO: Handle train, test, and validation splits properly (will we need unique index files?)
        # - Create a better solution for taking subsets of the data
        # - Eventually, determine how to sample specific subsets of the data that we care about.
        if n_items is not None:
            # Limit to n_items. Shuffle so each subset is random
            random.shuffle(self.indexes)
            self.indexes = self.indexes[:n_items]


    def decode_img(self, img):
        if not self.images:
            return np.array([])
        
        img_tensor = torch.from_numpy(np.frombuffer(img, dtype=np.uint8).copy()) 
        gpu_tensors_list = torchvision.io.decode_jpeg(
            img_tensor, 
            mode=torchvision.io.ImageReadMode.UNCHANGED,
            device= 'cuda' #['cuda:0', 'cuda:1'][torch.utils.data.get_worker_info().id%2]
        )
        # img_array = np.frombuffer(img, np.uint8)
        return gpu_tensors_list.cpu()
    
    def __len__(self):
        return len(self.indexes)
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            id, num_workers = worker.id, worker.num_workers
            local_indexes = []
            batch_id = 0
            for i in range(0, len(self.indexes), self.batch_size):
                if batch_id % num_workers == id:
                    local_indexes.extend(list(range(i,  min(len(self.indexes), i + self.batch_size))))
                
                batch_id += 1
        else:
            local_indexes = list(range(len(self.indexes)))    


        for idx in local_indexes:
            frame = e2e_pb2.E2EDFrame() # type: ignore
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


            yield {'PAST': past, 'FUTURE': future, 'IMAGES': [self.decode_img(images.image) for images in frame.frame.images], 'INTENT': frame.intent}

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    DATA_DIR = '/scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0/'
    # DATA_DIR = './data'
    # DATA_DIR = '/tmp/'
    BATCH_SIZE = 32
    dataset = WaymoE2E(BATCH_SIZE, data_dir = DATA_DIR, images=False)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=16,
    )
    
    def main():
        # start = time.time()
        for batch_of_frames in tqdm(loader):
            print(batch_of_frames.keys(), [b.shape for b in batch_of_frames.values() if isinstance(b, torch.Tensor)])
            pass
        # print("Total Time:", time.time()-start)
    
    import cProfile
    main()
