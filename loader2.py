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
import mmap
import cv2

class WaymoE2E(Dataset): 
    def __init__(self, indexFile = 'index.pkl', data_dir='./dataset'):
        self.DATA_DIR = data_dir

        with open(indexFile, 'rb') as f:
            self.indexes = pickle.load(f)
        
        all_paths = list(set([index[0] for index in self.indexes]))
        self.fds = [(os.open(os.path.join(self.DATA_DIR, fp), os.O_RDONLY), fp) for fp in all_paths]
        self.mmaps = {fd[1]: mmap.mmap(fd[0], length=0, access=mmap.ACCESS_READ) for fd in self.fds}

    def decode_img(self, img):
        img_array = np.frombuffer(img, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        frame = e2e_pb2.E2EDFrame()

        # self.mmaps[self.indexes[idx][0]].seek(self.indexes[idx][1])
        # frame.ParseFromString(self.mmaps[self.indexes[idx][0]].read(self.indexes[idx][2]))
        frame.ParseFromString(self.mmaps[self.indexes[idx][0]][self.indexes[idx][1]:self.indexes[idx][1]+self.indexes[idx][2]])

        return 1#np.vstack([self.decode_img(images.image) for images in frame.frame.images]), np.array(list(zip(frame.future_states.pos_x, frame.future_states.pos_y, frame.future_states.pos_z)))

from torch.utils.data import DataLoader
import time
from tqdm import tqdm
DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0'



def main():
    try:
        dataset = WaymoE2E(data_dir = DATA_DIR)

        loader = DataLoader(
            dataset, 
            batch_size=32,
            num_workers=8,
            shuffle=True
        )

        start = time.time()
        for batch_of_frames in tqdm(loader):
            pass
        print("Total Time:", time.time()-start)
    except Exception as e:
        print(e)
    finally:
        [mmap.close() for mmap in dataset.mmaps.values()]
        [os.close(fd[0]) for fd in dataset.fds]

import cProfile
# cProfile.run('main()')
main()