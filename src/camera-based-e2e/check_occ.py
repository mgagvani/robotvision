"""
OCC sanity checker — run from the camera-based-e2e root:

    python check_occ.py

Paths are taken directly from train_occ.slurm.
"""

import pickle
import os
import numpy as np

SCRIPT_DIR  = "/scratch/gilbreth/kumar753/robotvision/robotvision/src/camera-based-e2e"
OCC_ROOT    = "/scratch/gilbreth/kumar753/waymo_occ_new"
INDEX_TRAIN = os.path.join(SCRIPT_DIR, "index_train.pkl")
INDEX_VAL   = os.path.join(SCRIPT_DIR, "index_val.pkl")
N_SAMPLES   = 20


def load_indexes(pkl_path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return [(i, item) for i, item in enumerate(raw)]


def check_split(split, indexes, n_samples, label):
    print(f"\n{'='*60}")
    print(f"  {label}  ({split})  — checking {n_samples} of {len(indexes)} samples")
    print(f"{'='*60}")

    sample_idxs = list(range(min(n_samples, len(indexes))))
    missing, bad_shape, bad_dtype = [], [], []
    all_unique = set()

    for i in sample_idxs:
        orig_idx, _ = indexes[i]
        path = os.path.join(OCC_ROOT, split, f"occ_{orig_idx:07d}.npy")

        if not os.path.exists(path):
            missing.append((i, orig_idx, path))
            continue

        arr = np.load(path)

        if arr.shape != (100, 100, 16):
            bad_shape.append((i, orig_idx, arr.shape))

        if arr.dtype != np.uint8:
            bad_dtype.append((i, orig_idx, str(arr.dtype)))

        all_unique.update(arr.flatten().tolist())

    # ---- report ----
    print(f"\n[1] File existence")
    if missing:
        print(f"    MISSING {len(missing)} files!")
        for i, orig, p in missing[:5]:
            print(f"      sample[{i}] orig_idx={orig}  ->  {p}")
        if len(missing) > 5:
            print(f"      ... and {len(missing)-5} more")
    else:
        print(f"    OK — all {len(sample_idxs)} files found")

    print(f"\n[2] Shape  (expected (100, 100, 16))")
    if bad_shape:
        print(f"    BAD SHAPES on {len(bad_shape)} files!")
        for i, orig, sh in bad_shape[:5]:
            print(f"      sample[{i}] orig_idx={orig}  ->  shape={sh}")
    else:
        print(f"    OK — all shapes correct")

    print(f"\n[3] Dtype  (expected uint8)")
    if bad_dtype:
        print(f"    BAD DTYPE on {len(bad_dtype)} files!")
        for i, orig, dt in bad_dtype[:5]:
            print(f"      sample[{i}] orig_idx={orig}  ->  dtype={dt}")
    else:
        print(f"    OK — all uint8")

    print(f"\n[4] Unique values across sampled files")
    sorted_vals = sorted(all_unique)
    print(f"    {sorted_vals}")
    if 255 in all_unique:
        print(f"    (255 present — used as ignore_index in cross_entropy, OK)")
    class_vals = [v for v in sorted_vals if v != 255]
    n_classes = len(class_vals)
    print(f"    Class range: [{min(class_vals) if class_vals else 'N/A'}, {max(class_vals) if class_vals else 'N/A'}]  =>  {n_classes} distinct classes")
    if n_classes != 6:
        print(f"    WARNING: occ_head outputs 6 classes but ground truth has {n_classes} — mismatch!")
    else:
        print(f"    OK — matches occ_head output of 6 classes")

    print(f"\n[5] Cross-entropy smoke test")
    try:
        import torch
        import torch.nn.functional as F
        orig_idx, _ = indexes[0]
        path = os.path.join(OCC_ROOT, split, f"occ_{orig_idx:07d}.npy")
        if os.path.exists(path):
            arr = np.load(path).astype(np.int64)
            occ_gt = torch.from_numpy(arr).unsqueeze(0)   # (1, 100, 100, 16)
            pred   = torch.randn(1, 6, 100, 100, 16)      # matches occ_head output
            loss   = F.cross_entropy(pred, occ_gt, ignore_index=255)
            print(f"    OK — loss={loss.item():.4f}  (random pred, just checking shapes)")
        else:
            print(f"    SKIPPED — first file missing")
    except Exception as e:
        print(f"    ERROR: {e}")

    return len(missing), len(bad_shape), len(bad_dtype)


def main():
    print(f"\nOCC_ROOT    = {OCC_ROOT}")
    print(f"INDEX_TRAIN = {INDEX_TRAIN}")
    print(f"INDEX_VAL   = {INDEX_VAL}")

    train_idxs = load_indexes(INDEX_TRAIN)
    val_idxs   = load_indexes(INDEX_VAL)

    print(f"\n[6] Index alignment check")
    print(f"    train orig_idx range: 0 .. {len(train_idxs)-1}")
    print(f"    val   orig_idx range: 0 .. {len(val_idxs)-1}")
    print(f"    Both restart from 0 — occ files must live in separate subdirs:")
    print(f"      {OCC_ROOT}/train/occ_XXXXXXX.npy")
    print(f"      {OCC_ROOT}/val/occ_XXXXXXX.npy")
    train_dir = os.path.join(OCC_ROOT, "train")
    val_dir   = os.path.join(OCC_ROOT, "val")
    print(f"    train dir exists: {os.path.isdir(train_dir)}")
    print(f"    val   dir exists: {os.path.isdir(val_dir)}")

    tm, ts, td = check_split("train", train_idxs, N_SAMPLES, "TRAIN")
    vm, vs, vd = check_split("val",   val_idxs,   N_SAMPLES, "VAL")

    total_issues = tm + ts + td + vm + vs + vd
    print(f"\n{'='*60}")
    if total_issues == 0:
        print("  All checks passed!")
    else:
        print(f"  {total_issues} issue(s) found — see above")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
