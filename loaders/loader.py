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
import cv2 

class WaymoE2E(Dataset): 
    def __init__(self, indexFile = 'index_local.pkl', data_dir='./dataset'):
        self.DATA_DIR = data_dir

        with open(indexFile, 'rb') as f:
            self.indexes = pickle.load(f)

    def decode_img(self, img):
        img_array = np.frombuffer(img, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        filename, offset, size = self.indexes[idx]
        full_filepath = os.path.join(self.DATA_DIR, filename)


        with open(full_filepath, 'rb') as f:
            # pass
            f.seek(offset)
            # blenth = f.read(8)
            # proto_len = struct.unpack('q', blenth)[0]
            # f.read(4)
            protobuff = f.read(size)

        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(protobuff)
        return np.vstack([self.decode_img(images.image) for images in frame.frame.images]), np.array(list(zip(frame.future_states.pos_x, frame.future_states.pos_y, frame.future_states.pos_z)))

from torch.utils.data import DataLoader
import time
from tqdm import tqdm
# DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0'
DATA_DIR = './data'
# DATA_DIR = '/tmp/'


def main():
    dataset = WaymoE2E(data_dir = DATA_DIR)

    loader = DataLoader(
        dataset, 
        batch_size=32,
        num_workers=16,
        shuffle=False
    )

    start = time.time()
    for batch_of_frames in tqdm(loader):
        pass
    print("Total Time:", time.time()-start)

import cProfile
cProfile.run('main()')
# main()