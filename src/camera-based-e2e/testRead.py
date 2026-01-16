import os
import struct
import time
import pickle
from tqdm import tqdm
import sys, os
from protos import e2e_pb2

DATA_DIR = '/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0/'
indexes = []

with open(os.path.join(DATA_DIR, 'val_202504211843.tfrecord-00027-of-00093_.gstmp'), 'rb') as fp:
            offset = 0
            while True:
                raw_length = fp.read(8)
                if len(raw_length) < 8:
                    break
                record_len = struct.unpack('Q', raw_length)[0]  # unsigned

                fp.read(4)  # skip first CRC

                start_payload_offset = offset + 8 + 4  # where protobuf actually starts

                data = fp.read(record_len)
                if len(data) != record_len:
                    print(f"Incomplete record at {file}:{offset}")
                    break

                fp.read(4)  # skip last CRC

                frame = e2e_pb2.E2EDFrame()
                try:
                    frame.ParseFromString(data)
                except Exception as e:
                    print(f"length of indexes: {len(indexes)}")
                    print(f"Failed to parse frame at {file}:{offset} -> {e}")
                    offset += 8 + 4 + record_len + 4
                    continue

                if len(frame.preference_trajectories) and frame.preference_trajectories[0].preference_score != -1:
                    indexes.append((file, start_payload_offset, record_len))

                # update offset for next record
                offset += 8 + 4 + record_len + 4