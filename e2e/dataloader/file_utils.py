import fcntl
import glob
import os
import shutil
import struct
import time

from tqdm import tqdm

from typing import Literal

IN_PROGRESS_SUFFIX = '.part'
LOCK_SUFFIX = '.lock'

def purge_from_tmp(tmp_path: str) -> None:
    """
    Remove a file from the temporary directory.

    Args:
        tmp_path (str): Path to a file in the temporary directory.

    """
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

        if os.path.exists(tmp_path + LOCK_SUFFIX):
            os.remove(tmp_path + LOCK_SUFFIX)

        if os.path.exists(tmp_path + IN_PROGRESS_SUFFIX):
            os.remove(tmp_path + IN_PROGRESS_SUFFIX)

def copy_to_tmp(source_path: str, tmp_path: str) -> str|None:
    """
    Copy a file to a temporary directory if it does not already exist there.

    Args:
        source_path (str): Path to the source file.
        tmp_path (str): Path to the temporary directory.

    Returns:
        str: Path to the copied file in the temporary directory.
    """
    try:
        os.makedirs(tmp_path, exist_ok=True)
    except OSError:
        # If tmp dir can't be created (e.g., no space), fall back to source.
        return None

    filename = os.path.basename(source_path)
    dest_path = os.path.join(tmp_path, filename)

    part_path = dest_path + IN_PROGRESS_SUFFIX
    lock_path = dest_path + LOCK_SUFFIX

    if os.path.exists(dest_path) or os.path.exists(part_path) or os.path.exists(lock_path):
        return None


    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        # If tmp is full (ENOSPC) or inode exhausted, skip caching.
        return None
    with os.fdopen(lock_fd, 'r+') as lock_file:

        lock_fd = None
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Re-check under the lock
        if os.path.exists(dest_path):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return None
        
        # print(f'Copying {source_path} to tmp {dest_path}...')

        try:
            shutil.copy2(source_path, part_path)
            os.replace(part_path, dest_path)
            return dest_path
        except OSError as err:
            # If tmp is full (ENOSPC) or similar, clean partial and fall back to source.
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
            return None

    return None

def wait_for_tmp_file(basename: str, tmp_path: str, check_interval: float = 0.2) -> str:
    """
    Wait for a file to appear in the temporary directory.

    Args:
        basename (str): Basename of the file to wait for.
        tmp_path (str): Path to the temporary directory.
        check_interval (float): Time in seconds between checks.

    Returns:
        str: Path to the file in the temporary directory.
    """

    dest_path = os.path.join(tmp_path, basename)

    while not os.path.exists(dest_path):
        time.sleep(check_interval)

    return dest_path

def get_tf_filepaths(file_path: str, type: Literal['training', 'test', 'val']) -> list[str]:
    """
    Get TensorFlow Record filenames from a directory based on the specified type.

    Args:
        file_path (str): Directory path containing the TFRecord files.
        type (Literal['training', 'test', 'val']): Type of data to retrieve.

    Returns:
        list[str]: Sorted list of TFRecord file paths.
    """

    return sorted(glob.glob(os.path.join(file_path, f'{type}_*.tfrecord*')))

def index_tf_records(filepaths: list[str], out_path: str, show_progress = True, tmp_path = None) -> None:
    """
    Index TensorFlow Record files for efficient data loading.

    Args:
        filepaths (list[str]): List of paths to TensorFlow Record files.
        out_path (str): Path to save the index file.
    """

    if show_progress:
        wrap = tqdm
    else:
        wrap = lambda x: x

    index_file = open(out_path, 'wb')
    for i, file_path in wrap(enumerate(filepaths)):
        if tmp_path is not None:
            copied = copy_to_tmp(file_path, tmp_path)
            if copied is not None:
                file_path = copied
            else:
                # File already present in tmp
                file_path = os.path.join(tmp_path, os.path.basename(file_path))
        f = open(file_path, 'rb')

        while True:
            proto_size_bytes = f.read(8)
            if not proto_size_bytes:
                break  # EOF

            proto_size = struct.unpack('<Q', proto_size_bytes)[0]

            index_file.write(struct.pack('<Q', i))
            index_file.write(struct.pack('<Q', f.tell() + 4)) # +4 to skip length Checksum
            index_file.write(struct.pack('<Q', proto_size))

            # Move past proto and two checksums
            f.seek(proto_size + 8, os.SEEK_CUR)
        
        f.close()
        
        if tmp_path is not None:
            purge_from_tmp(file_path)
    
    index_file.close()

def load_index(index_path: str) -> list[tuple[int, int, int]]:
    """
    Load an index file created by index_tf_records.

    Args:
        index_path (str): Path to the index file.

    Returns:
        list[tuple[int, int, int]]: List of tuples containing (file_index, offset, size).
    """
    
    index = []

    f = open(index_path, 'rb')

    while True:
        file_idx_bytes = f.read(8)

        if not file_idx_bytes:
            break # EOF

        file_idx = struct.unpack('<Q', file_idx_bytes)[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        size = struct.unpack('<Q', f.read(8))[0]
        index.append((file_idx, offset, size))

    f.close()

    return index