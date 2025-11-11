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
from concurrent.futures import ThreadPoolExecutor

devices = ['cuda:0', 'cuda:1']

class WaymoE2E(IterableDataset): 
    def __init__(self, batch_size, indexFile = 'index.pkl', data_dir='./dataset', temp_dir='/tmp/cache/', precopy_factor=1, decode_image=True):
        self.precopy_factor = precopy_factor
        self.decode_image = decode_image
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.temp_dir = temp_dir
        
        self.copy_executor = ThreadPoolExecutor(max_workers=precopy_factor)

        with open(indexFile, 'rb') as f:
            self.indexes = pickle.load(f)

        self.all_filenames = []
        for index in self.indexes:
            if index[0] not in self.all_filenames:
                self.all_filenames.append(index[0])
        
        self.current_filename_index = 0
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.copy_futures_map = {} 
        self.copy_lock = threading.Lock() 

    def decode_img(self, img):
        # return img
        if not self.decode_image:
            return np.array([])
        img_tensor = torch.from_numpy(np.frombuffer(img, dtype=np.uint8).copy()) 
        gpu_tensors_list = torchvision.io.decode_jpeg(
            img_tensor, 
            mode=torchvision.io.ImageReadMode.UNCHANGED,
            device= 'cuda' #['cuda:0', 'cuda:1'][torch.utils.data.get_worker_info().id%2]
        )
        return gpu_tensors_list.cpu()
    
    def __len__(self):
        return len(self.indexes)
    
    def _copy_files(self, filenames):
        """Submits file copy tasks to the thread pool and stores the Futures in a map."""
        with self.copy_lock:
            finished_keys = [k for k, v in self.copy_futures_map.items() if v.done()]
            for k in finished_keys:
                del self.copy_futures_map[k]
            
            active_futures = len(self.copy_futures_map)
            
            for filename in filenames:
                if filename not in self.copy_futures_map:
                    if active_futures < self.precopy_factor:
                        future = self.copy_executor.submit(self._copy_file, filename)
                        self.copy_futures_map[filename] = future 
                        active_futures += 1
                    else:
                        break 
    
    def _copy_file(self, filename):
        source_path, dest_path = os.path.join(self.data_dir, filename), os.path.join(self.temp_dir, filename)
        temp_dest_path = dest_path + f".tmp.{os.getpid()}" 
        lock_path = dest_path + ".lock"

        if os.path.exists(dest_path):
            return dest_path
    
        lock_file = open(lock_path, 'w')
        
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            if os.path.exists(dest_path):
                return dest_path
            
            try:
                start = time.time()
                shutil.copy(source_path, temp_dest_path)
                print(f"Copy took {time.time()-start} seconds for {filename}")
                os.rename(temp_dest_path, dest_path)
                return dest_path
                
            finally:
                if os.path.exists(temp_dest_path):
                    os.remove(temp_dest_path)
        except Exception as e:
            print(f"Error during copy of {filename}: {e}") 
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()

        worker_file_set = set()
        if worker is not None:
            id, num_workers = worker.id, worker.num_workers
            local_indexes = []
            batch_id = 0
            for i in range(0, len(self.indexes), self.batch_size):
                if batch_id % num_workers == id:
                    local_indexes.extend(list(range(i, min(len(self.indexes), i + self.batch_size))))
                batch_id += 1
        else:
            local_indexes = list(range(len(self.indexes)))
            
        for idx in local_indexes:
            worker_file_set.add(self.indexes[idx][0])

        worker_files = sorted(list(worker_file_set), key=lambda f: self.all_filenames.index(f))
        
        if not worker_files:
            return iter([])

        self.file = None
        self.filename = None

        current_file_idx_in_all = self.all_filenames.index(worker_files[0])
        
        initial_copy_list = self.all_filenames[current_file_idx_in_all : current_file_idx_in_all + self.precopy_factor]
        self._copy_files(initial_copy_list)
        

        for idx in local_indexes:
            frame = e2e_pb2.E2EDFrame() # type: ignore

            filename, start_byte, byte_length = self.indexes[idx]
            
            if self.filename != filename:
                if self.file:
                    self.file.close()
                
                try:
                    future_to_wait_on = None
                    with self.copy_lock:
                        future_to_wait_on = self.copy_futures_map.get(filename)
                    
                    if future_to_wait_on:
                        future_to_wait_on.result() 
                    else:
                        self._copy_file(filename) 
                        
                except Exception as e:
                    print(f"Error waiting for or copying file {filename}: {e}")
                    continue 

                self.file = open(os.path.join(self.temp_dir, filename), 'rb')
                self.filename = filename
                
                current_file_global_idx = self.all_filenames.index(self.filename)
                next_file_global_idx = current_file_global_idx + 1
                
                if next_file_global_idx < len(self.all_filenames):
                    files_to_precopy = self.all_filenames[next_file_global_idx : next_file_global_idx + self.precopy_factor]
                    self._copy_files(files_to_precopy)
            
            self.file.seek(start_byte) # type: ignore
            protobuf = self.file.read(byte_length) # type: ignore
            frame.ParseFromString(protobuf)

            yield [self.decode_img(images.image) for images in frame.frame.images], np.stack((frame.future_states.pos_x, frame.future_states.pos_y, frame.future_states.pos_z), axis=-1)

        if self.file:
            self.file.close()


DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0'
# DATA_DIR = './data'
BATCH_SIZE = 32
PRECOPY_FACTOR = 4
NUM_WORKERS = 8

dataset = WaymoE2E(BATCH_SIZE, data_dir = DATA_DIR, precopy_factor=PRECOPY_FACTOR, decode_image=False)
loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

def main():
    # start = time.time()
    for batch_of_frames in tqdm(loader):
        pass
    # print("Total Time:", time.time()-start)

import cProfile
cProfile.run('main()')
# main()