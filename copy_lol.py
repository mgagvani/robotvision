import shutil
import time
import os
DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0/training_202504031202_202504151040.tfrecord-00111-of-00263'
DEST = '/tmp/cache/'
if not os.path.exists(DEST):
    os.mkdir(DEST)
start=time.time()
try:
    shutil.copy(DATA_DIR, DEST)
except FileExistsError as e:
    pass
print('Total time', time.time()-start)