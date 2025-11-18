import torch
from torch.utils.data import IterableDataset, DataLoader
from protos import e2e_pb2
import torchvision
import pickle
import shutil
import os
import numpy as np
import threading
import fcntl
import time
from tqdm import tqdm

devices = ['cuda:0', 'cuda:1']

class WaymoE2E(IterableDataset): 
    def __init__(self, batch_size, indexFile = 'index.pkl', data_dir='./dataset', temp_dir='/tmp/cache/', precopy_factor=1, decode_image=True):
        self.precopy_factor=precopy_factor
        self.decode_image = decode_image
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.temp_dir = temp_dir


        with open(indexFile, 'rb') as f:
            self.indexes = pickle.load(f)[:20000]

        self.filenames = []
        for index in self.indexes:
            if index[0] not in self.filenames:
                self.filenames.append(index[0])
        self.filename = self.filenames.pop(0)
        os.makedirs(self.temp_dir, exist_ok=True)


    def decode_img(self, img):
        # return img
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

    def _copy_file(self, filename):
        source_path, dest_path = os.path.join(self.data_dir, filename), os.path.join(self.temp_dir, filename)
        temp_dest_path = dest_path + f".tmp.{os.getpid()}" #prevents the file from being read before it is copied
        lock_path = dest_path + ".lock"

        if os.path.exists(dest_path):
            return
    
        lock_file = open(lock_path, 'w')
        
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            if os.path.exists(dest_path):
                return dest_path
            
            try:
                start = time.time()
                shutil.copy(source_path, temp_dest_path)
                print(f"Copy took {time.time()-start} seconds")
                os.rename(temp_dest_path, dest_path)
                
            finally:
                if os.path.exists(temp_dest_path):
                    os.remove(temp_dest_path)
        except Exception as e:
            print("waiting")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()

        self._copy_file(self.filename)
        self.file = open(os.path.join(self.temp_dir, self.filename), 'rb')
        
        self.next_filename = self.filenames.pop(0)
        
        self.copy = threading.Thread(target=self._copy_file, name="copy file", args=(self.next_filename,))
        self.copy.daemon = True
        self.copy.start()

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
                
                self.copy.join()
                self.file = open(os.path.join(self.temp_dir, self.next_filename), 'rb')
                self.filename = self.next_filename
                
                if len(self.filenames):
                    self.next_filename = self.filenames.pop(0)
                    self.copy = threading.Thread(target=self._copy_file, name="copy file", args=(self.next_filename,))
                    self.copy.daemon = True
                    self.copy.start()

            self.file.seek(start_byte) # type: ignore
            protobuf = self.file.read(byte_length) # type: ignore
            frame.ParseFromString(protobuf)

            yield [self.decode_img(images.image) for images in frame.frame.images], np.stack((frame.future_states.pos_x, frame.future_states.pos_y, frame.future_states.pos_z), axis=-1)


DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0'
# DATA_DIR = './data'
# DATA_DIR = '/tmp/'
BATCH_SIZE = 32
dataset = WaymoE2E(BATCH_SIZE, data_dir = DATA_DIR)
loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE,
    num_workers=12,
)

def main():
    # start = time.time()
    for batch_of_frames in tqdm(loader):
        pass
    # print("Total Time:", time.time()-start)

import cProfile
main()