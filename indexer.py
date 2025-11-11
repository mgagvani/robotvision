should_index = True
if should_index: # Takes approx. 9 mins to index
    import os
    import mmap
    import struct
    import time
    import pickle
    from tqdm import tqdm

    DATA_DIR = '/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0'
    # DATA_DIR = './data'

    indexes = []

    start = time.time()
    for i, fn in enumerate(tqdm([file for file in os.listdir(DATA_DIR) if '.tfrecord' in file and file.startswith('train')])):
        with open(os.path.join(DATA_DIR, fn), 'rb') as file:
            while True:
                blenth = file.read(8)
                if len(blenth) == 0:
                    break
                proto_len = struct.unpack('q', blenth)[0]
                indexes.append((fn, file.tell()+4, proto_len)) #file.tell()+4 so that I can make sure we skip checksum
                file.seek(file.tell() + proto_len+8)


    with open('index.pkl', 'wb') as f:
        pickle.dump(indexes, f)

    print("Done indexing")
