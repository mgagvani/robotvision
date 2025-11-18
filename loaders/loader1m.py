# HISTORICAL, loader1-3 have memory leaks and issues
import torch
from torch.utils.data import Dataset
from protos import e2e_pb2
import pickle
import struct
import os
import numpy as np
from PIL import Image
from io import BytesIO 

class WaymoE2E(Dataset): 
    def __init__(self, indexFile='index.pkl', data_dir='./dataset'):
        self.DATA_DIR = data_dir

        with open(indexFile, 'rb') as f:
            self.indexes = pickle.load(f)


    def decode_img(self, img):
        return np.array(Image.open(BytesIO(img)))
        
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        filename, offset, size = self.indexes[idx]
        full_filepath = os.path.join(self.DATA_DIR, filename)

        with open(full_filepath, 'rb') as f:
            f.seek(offset)
            protobuff = f.read(size)

        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(protobuff)

        return 1 # np.vstack([self.decode_img(images.image) for images in frame.frame.images]), np.array(list(zip(frame.future_states.pos_x, frame.future_states.pos_y, frame.future_states.pos_z)))

from torch.utils.data import DataLoader
import time
from tqdm import tqdm
DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0'

dataset = WaymoE2E(data_dir = DATA_DIR)

loader = DataLoader(
    dataset, 
    batch_size=32,
    num_workers=8,
    shuffle=False
)

start = time.time()
for batch_of_frames in tqdm(loader):
    pass
print("Total Time:", time.time()-start)
