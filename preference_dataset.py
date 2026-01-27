import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import e2e_pb2
from tqdm import tqdm
import random


class PreferenceDataset(Dataset):    
    def __init__(self, index_file, data_dir, cache_file='preference_cache_full.pkl'):
        self.data_dir = data_dir

        if os.path.exists(cache_file):
            print(f"Loading filtered samples from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.valid_samples = pickle.load(f)
            print(f"Loaded {len(self.valid_samples)} valid preference trajectory samples from cache")
            return
        
        print("Cache not found. Filtering dataset for valid preference trajectories...")
        with open(index_file, 'rb') as f:
            all_indexes = pickle.load(f)
        
        self.valid_samples = []
        
        # Group indexes by filename
        file_groups = {}
        for idx, (filename, start_byte, byte_length) in enumerate(all_indexes):
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append((idx, start_byte, byte_length))
        
        # Process each file once
        for filename, frame_list in tqdm(file_groups.items(), desc="Processing files"):
            file_path = os.path.join(data_dir, filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"WARNING: File not found: {file_path}")
                continue
                
            with open(file_path, 'rb') as f:
                for original_idx, start_byte, byte_length in frame_list:
                    # Read and parse frame
                    f.seek(start_byte)
                    protobuf = f.read(byte_length)
                    frame = e2e_pb2.E2EDFrame()
                    frame.ParseFromString(protobuf)
                    print("num preference trajectories:", len(frame.preference_trajectories))
                    
                    # Check if this frame has valid preference trajectories
                    for traj_idx, pref_traj in enumerate(frame.preference_trajectories):
                        if pref_traj.preference_score >= 0:  # Valid score
                            self.valid_samples.append({
                                'file_info': (filename, start_byte, byte_length),
                                'traj_idx': traj_idx,
                                'original_idx': original_idx
                            })
        
        print(f"Found {len(self.valid_samples)} valid preference trajectory samples")
        
        # Save to cache
        print(f"Saving filtered samples to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.valid_samples, f)
        print("Cache saved!")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample_info = self.valid_samples[idx]
        filename, start_byte, byte_length = sample_info['file_info']
        traj_idx = sample_info['traj_idx']
        
        # Read frame
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'rb') as f:
            f.seek(start_byte)
            protobuf = f.read(byte_length)
            frame = e2e_pb2.E2EDFrame()
            frame.ParseFromString(protobuf)
        
        # Extract past states (T,6)
        past = np.stack([
            frame.past_states.pos_x,
            frame.past_states.pos_y,
            frame.past_states.vel_x,
            frame.past_states.vel_y,
            frame.past_states.accel_x,
            frame.past_states.accel_y
        ], axis=-1).astype(np.float32)
        
        # Extract intent (one-hot encode)
        intent = frame.intent
        intent_onehot = np.zeros(4, dtype=np.float32)
        intent_onehot[intent] = 1.0
        
        # Extract preference trajectory (only pos_x, pos_y are populated)
        pref_traj = frame.preference_trajectories[traj_idx]
        trajectory = np.stack([
            pref_traj.pos_x,
            pref_traj.pos_y
        ], axis=-1).astype(np.float32)
        
        # Extract preference score (target)
        score = pref_traj.preference_score
        
        return {
            'past_states': torch.from_numpy(past.flatten()),
            'intent': torch.from_numpy(intent_onehot),
            'trajectory': torch.from_numpy(trajectory.flatten()),
            'score': torch.tensor(score, dtype=torch.float32)
        }
