"""Extract and save camera images from a Waymo E2E frame by index.

Usage:
    python extract_frame_images.py --idx 1
"""
import argparse
import os
import pickle
import cv2
import numpy as np
from pathlib import Path

from protos import e2e_pb2

PARENT_DIR = "/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e"
DATA_DIR   = "/scratch/gilbreth/kumar753/robotvision/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="./frame_images")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    index_file = os.path.join(PARENT_DIR, "index_train.pkl")
    with open(index_file, "rb") as f:
        indexes = pickle.load(f)

    filename, start_byte, byte_length = indexes[args.idx]
    with open(os.path.join(DATA_DIR, filename), "rb") as f:
        f.seek(start_byte)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(f.read(byte_length))

    print(f"Frame {args.idx}: {len(frame.frame.images)} cameras")

    for img_proto in frame.frame.images:
        cam_name = img_proto.name
        jpg = np.frombuffer(img_proto.image, dtype=np.uint8)
        rgb = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        out_path = os.path.join(args.out_dir, f"frame_{args.idx:07d}_cam{cam_name}.jpg")
        cv2.imwrite(out_path, rgb)
        print(f"  Saved: {out_path}")

if __name__ == "__main__":
    main()
