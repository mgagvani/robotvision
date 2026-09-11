"""Index WOD-E2E TFRecords so loader.py can seek straight to each frame.

    python build_index.py /path/to/waymo_open_dataset_end_to_end_camera_v_1_0_0

Writes index_{train,val,test}.pkl next to this file: lists of
(tfrecord filename, byte offset of the E2EDFrame proto, proto length).
Takes roughly 10 minutes. The repository already ships the three index files
for WOD-E2E v1.0.0, so this only needs to run for a fresh dataset version.
"""
import os
import pickle
import struct
import sys

from tqdm import tqdm

DATA_DIR = sys.argv[1]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

for category in ["train", "val", "test"]:
    indexes = []
    files = sorted(f for f in os.listdir(DATA_DIR) if ".tfrecord" in f and f.startswith(category))
    for fn in tqdm(files, desc=category):
        with open(os.path.join(DATA_DIR, fn), "rb") as file:
            while True:
                header = file.read(8)
                if len(header) == 0:
                    break
                proto_len = struct.unpack("q", header)[0]
                # +4 skips the length CRC; the record is followed by a 4-byte data CRC.
                indexes.append((fn, file.tell() + 4, proto_len))
                file.seek(file.tell() + proto_len + 8)
    with open(os.path.join(OUT_DIR, f"index_{category}.pkl"), "wb") as f:
        pickle.dump(indexes, f)
print("Done indexing")
