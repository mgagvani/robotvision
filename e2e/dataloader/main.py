
import os
from tqdm import tqdm

from torch.utils.data import DataLoader

import dataloader.file_utils as file_utils
from dataloader.dataset import WaymoDataset

DATA_DIR = '/scratch/gilbreth/peter922/waymo-data/'
INDEX_FILENAME = 'index.idx'
TMP_DIR = os.environ.get('TMPDIR', '/tmp/')
# TMP_DIR = None

def main():

    index_path = os.path.join(DATA_DIR, INDEX_FILENAME)

    train_file_paths = file_utils.get_tf_filepaths(DATA_DIR, 'training')

    print ('Indexing TFRecords if needed...')
    if not os.path.exists(index_path):
        print('Index file not found, indexing now...')
        file_utils.index_tf_records(train_file_paths, index_path, True, TMP_DIR)
    
    print('Done indexing. Creating dataset...')

    index = file_utils.load_index(index_path)

    train_dataset = WaymoDataset(
        data_root=DATA_DIR,
        file_paths=train_file_paths,
        index=index,
        num_items=None,
        tmp_path=TMP_DIR,
    )

    # for images, future_states in tqdm(train_dataset):
    #     pass

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=8
    )

    print('Dataset created, iterating through data loader...')

    for images, future_states in tqdm(loader):
        pass



if __name__ == "__main__":
    main()