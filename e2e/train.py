import argparse
from datetime import datetime
import os
from pathlib import Path

from e2e.models.models import BaseModel, LitModel, MonocularModel
from e2e.models.features import VitFeatures
from e2e.dataloader.dataset import WaymoDataset, gpu_decode_collate_fn
import e2e.dataloader.file_utils as file_utils

import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

DATA_DIR = '/scratch/gilbreth/peter922/waymo-data/'
TRAIN_INDEX_FILENAME = 'train_index.idx'
VAL_INDEX_FILENAME = 'val_index.idx'
TMP_DIR = os.environ.get('TMPDIR', '/tmp/')

# default cam set
DEFAULT_CAM_IDS = [1, 2, 3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to Waymo E2E data directory')
    parser.add_argument('--tmp_dir', type=str, default=TMP_DIR, help='Path to temporary directory for data loading')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--idx_only', action='store_true', help='Only create index files and exit')
    parser.add_argument('--cam_ids', type=str, default=str(DEFAULT_CAM_IDS),
                        help='Comma-separated list or Python-style list of camera ids. Interpreted as CameraName enum values (e.g., "1,2,3" for FRONT/FRONT_LEFT/FRONT_RIGHT).')
    args = parser.parse_args()

    # Parse camera ids
    cam_ids_str = args.cam_ids.strip()
    if cam_ids_str.startswith('['):
        import ast
        cam_ids = ast.literal_eval(cam_ids_str)
    elif cam_ids_str == '':
        cam_ids = []
    else:
        cam_ids = [int(x) for x in cam_ids_str.split(',') if x.strip()]

    train_files = file_utils.get_tf_filepaths(args.data_dir, 'training')
    val_files = file_utils.get_tf_filepaths(args.data_dir, 'val')

    print("Using cam_ids:", cam_ids)

    if args.idx_only:
        # Remove old index files if they exist
        train_index_path = os.path.join(args.data_dir, TRAIN_INDEX_FILENAME)
        val_index_path = os.path.join(args.data_dir, VAL_INDEX_FILENAME)
        if os.path.exists(train_index_path):
            os.remove(train_index_path)
        if os.path.exists(val_index_path):  
            os.remove(val_index_path)

    # Make index files if needed
    if args.idx_only or not os.path.exists(os.path.join(args.data_dir, TRAIN_INDEX_FILENAME)):
        # file_utils.create_data_index(train_files, os.path.join(args.data_dir, TRAIN_INDEX_FILENAME))
        file_utils.index_tf_records(train_files, os.path.join(args.data_dir, TRAIN_INDEX_FILENAME), True, args.tmp_dir)

    if args.idx_only or not os.path.exists(os.path.join(args.data_dir, VAL_INDEX_FILENAME)):
        # file_utils.create_data_index(val_files, os.path.join(args.data_dir, VAL_INDEX_FILENAME)) 
        file_utils.index_tf_records(val_files, os.path.join(args.data_dir, VAL_INDEX_FILENAME), True, args.tmp_dir)
    if args.idx_only:
        print("Index files created. Exiting as --idx_only was set.")
        return

    # Data 
    # TODO - make this use a proper train / val split, and to sample only specific data that is long-tailed (e.g., difficult and out of distribution)
    
    # train_dataset = WaymoDataset(DATA_DIR, file_utils.get_tf_filepaths(DATA_DIR, 'training'), file_utils.load_index(os.path.join(DATA_DIR, INDEX_FILENAME)), 1000, TMP_DIR)
    # val_dataset = WaymoDataset(DATA_DIR, file_utils.get_tf_filepaths(DATA_DIR, 'training'), file_utils.load_index(os.path.join(DATA_DIR, INDEX_FILENAME)), 300, TMP_DIR)

    train_dataset = WaymoDataset(
        args.data_dir, 
        train_files, 
        file_utils.load_index(os.path.join(args.data_dir, TRAIN_INDEX_FILENAME)), 
        # 60000, 
        None,
        args.tmp_dir,
        image_ids=cam_ids
    )

    val_dataset = WaymoDataset(
        args.data_dir, 
        val_files, 
        file_utils.load_index(os.path.join(args.data_dir, VAL_INDEX_FILENAME)), 
        60000,
        # None,
        args.tmp_dir,
        image_ids=cam_ids
    )

    torch.set_float32_matmul_precision('medium')

    mp_context = mp.get_context("spawn")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        persistent_workers=True,
        collate_fn=gpu_decode_collate_fn,
        multiprocessing_context=mp_context,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        persistent_workers=True,
        collate_fn=gpu_decode_collate_fn,
        multiprocessing_context=mp_context,
    )

    # Model
    in_dim = 16 * 6  # Past: (B, 16, 6)
    out_dim = 20 * 2  # Future: (B, 20, 2)

    model = MonocularModel(
        in_dim=in_dim,
        out_dim=out_dim,
        feature_extractor=VitFeatures(),
        num_cams=len(cam_ids),
    )

    # lit_model = LitModel(model=torch.compile(model, mode='max-autotune'), lr=args.lr)
    lit_model = LitModel(model=model, lr=args.lr)

    base_path = Path(args.data_dir).parent.as_posix()

    early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=1e-3)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
        # log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=CSVLogger(os.path.join(base_path, "logs"), name=f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"),
        # precision="bf16-mixed",
        # profiler="simple",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            early_stop
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
