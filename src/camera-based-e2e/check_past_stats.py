"""
Check past state statistics from the dataset.
Run from camera-based-e2e root:

    python check_past_stats.py
"""

import pickle
import os
import numpy as np
import sys

sys.path.insert(0, "/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e")

from protos import e2e_pb2

INDEX_FILE = "/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e/index_train.pkl"
DATA_DIR   = "/scratch/gilbreth/kumar753/robotvision/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0"
N_SAMPLES  = 200

FEATURE_NAMES = ["pos_x", "pos_y", "vel_x", "vel_y", "accel_x", "accel_y"]

def main():
    with open(INDEX_FILE, "rb") as f:
        indexes = pickle.load(f)

    print(f"Checking {N_SAMPLES} samples from {INDEX_FILE}\n")

    all_past = []
    current_file, fh = None, None

    for i in range(min(N_SAMPLES, len(indexes))):
        filename, start_byte, byte_length = indexes[i]
        path = os.path.join(DATA_DIR, filename)

        if current_file != path:
            if fh:
                fh.close()
            fh = open(path, "rb")
            current_file = path

        fh.seek(start_byte)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(fh.read(byte_length))

        past = np.stack([
            frame.past_states.pos_x,
            frame.past_states.pos_y,
            frame.past_states.vel_x,
            frame.past_states.vel_y,
            frame.past_states.accel_x,
            frame.past_states.accel_y,
        ], axis=-1).astype(np.float32)  # (16, 6)

        all_past.append(past)

    if fh:
        fh.close()

    all_past = np.stack(all_past, axis=0)  # (N, 16, 6)

    print(f"{'Feature':<12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 55)
    for i, name in enumerate(FEATURE_NAMES):
        col = all_past[:, :, i]
        print(f"{name:<12} {col.mean():>10.3f} {col.std():>10.3f} {col.min():>10.3f} {col.max():>10.3f}")

    print("\nOverall (all features):")
    print(f"  mean={all_past.mean():.3f}  std={all_past.std():.3f}  min={all_past.min():.3f}  max={all_past.max():.3f}")

    print("\nDiagnosis:")
    pos = all_past[:, :, :2]
    if np.abs(pos).max() > 50:
        print("  WARNING: pos_x/pos_y are in absolute world coordinates (large values).")
        print("           If your supervisor normalized these, that explains the ADE gap.")
        print("           Consider subtracting the last observed position:")
        print("             past[:, :, :2] -= past[:, -1:, :2]")
    else:
        print("  OK: positions look relative/small — probably already normalized.")

    vel = all_past[:, :, 2:4]
    if np.abs(vel).max() > 50:
        print("  WARNING: vel_x/vel_y have very large values — check units (m/s vs km/h?).")
    else:
        print("  OK: velocities look reasonable.")

if __name__ == "__main__":
    main()
