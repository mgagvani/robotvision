import os
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import torch
import torchvision


from e2e.protos import e2e_pb2
import e2e.dataloader.file_utils as file_utils

def gpu_decode_collate_fn(batch):
    """
    Collate function to decode images on GPU during data loading.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        dict: Batch with decoded images and other data.
    """
    past = torch.stack([item['past'] for item in batch], dim=0)
    future = torch.stack([item['future'] for item in batch], dim=0)
    intent = torch.tensor([item['intent'] for item in batch], dtype=torch.long)

    device = torch.device('cuda:0')

    # Decode images on GPU
    decoded_images = []
    for item in batch:
        images = []
        for img_bytes in item['image_bytes']:

            # decode_jpeg expects CPU input; copy buffer to writable CPU tensor, then place output on device
            cpu_tensor = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.uint8).copy())

            if torch.isnan(cpu_tensor).any() or torch.isinf(cpu_tensor).any():
                raise ValueError("Image byte tensor contains NaN or Inf values.")

            img = torchvision.io.decode_jpeg(cpu_tensor, device=device)

            if torch.isnan(img).any() or torch.isinf(img).any():
                raise ValueError("Decoded image contains NaN or Inf values.")
            
            # Force a common spatial size so stacking across cameras works even if raw resolutions differ.
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0).float(), size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze(0)

            if torch.isnan(img).any() or torch.isinf(img).any():
                raise ValueError("Resized image contains NaN or Inf values.")

            images.append(img)

        if images:
            decoded_images.append(torch.stack(images, dim=0))  # Stack images for this sample
        else:
            # No cameras provided; skip image stacking
            decoded_images.append(None)

    if any(x is None for x in decoded_images):
        images_batch = None
    else:
        images_batch = torch.stack(decoded_images, dim=0)  # Stack all samples

    return {
        'past': past,
        'future': future,
        'images': images_batch,
        'intent': intent
    }

class WaymoDataset(IterableDataset):
    def __init__(self, data_root, file_paths, index, num_items=None, tmp_path = None, pre_copy_factor = 5, image_ids=[], max_tmp_files=20):
        self.data_root = data_root
        self.file_paths = file_paths

        if num_items is None:
            self.index = index
        else:
            self.index = self.__select_random_sub_index(index, num_items)

        self.tmp_path = tmp_path
        self.pre_copy_factor = pre_copy_factor
        self.copied_file_ids = []
        self.max_tmp_files = max_tmp_files
        self.image_ids = image_ids

    def __enforce_tmp_limit(self):
        while len(self.copied_file_ids) > self.max_tmp_files:
            file_id = self.copied_file_ids.pop(0)

            tmp_file_path = os.path.join(self.tmp_path, os.path.basename(self.file_paths[file_id]))

            try:
                file_utils.purge_from_tmp(tmp_file_path)
            except Exception as e:
                print(f"Warning: Failed to purge temp file {tmp_file_path}: {e}")

    def __select_random_sub_index(self, index: list, num_items: int) -> np.ndarray:
        
        if num_items >= len(index):
            return np.random.permutation(index).tolist()
        
        return np.random.permutation(index)[:num_items].tolist()

    def __shard_index(self):
        wi = get_worker_info()
        if wi is None:
            return self.index
        
        return [rec for i, rec in enumerate(self.index) if i % wi.num_workers == wi.id]
    
    def __decode_image(self, img):
        tensor = torch.from_numpy(np.frombuffer(img, dtype=np.uint8).copy())

        return torchvision.io.decode_jpeg(
            tensor,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )

    def __iter__(self):
        """
        
        Currently includes:
        - All encoded images
        - Future states
        
        """


        shard = self.__shard_index()

        for file_idx, offset, size in shard:
            file_path = self.file_paths[file_idx]

            # Pre-copy files to tmp_path if needed
            if self.tmp_path is not None:
                # Purge before copying to keep space available.
                self.__enforce_tmp_limit()

                for j in range(file_idx, file_idx + 1 + self.pre_copy_factor):
                    if j >= len(self.file_paths):
                        break
                    if j in self.copied_file_ids:
                        continue

                    dst = file_utils.copy_to_tmp(self.file_paths[j], self.tmp_path)
                    if dst is not None:
                        # This thread actually copied the file, it wasn't taken by another thread.
                        self.copied_file_ids.append(j)

                # Use tmp copy if present; otherwise fall back to source file.
                candidate = os.path.join(self.tmp_path, os.path.basename(self.file_paths[file_idx]))
                if os.path.exists(candidate):
                    # Ensure the cached copy matches the source size; stale partials can corrupt offsets.
                    try:
                        src_size = os.path.getsize(self.file_paths[file_idx])
                        dst_size = os.path.getsize(candidate)
                        if src_size != dst_size:
                            file_utils.purge_from_tmp(candidate)
                            copied = file_utils.copy_to_tmp(self.file_paths[file_idx], self.tmp_path)
                            if copied is not None:
                                candidate = copied
                    except OSError:
                        # If size check fails, fall back to source below.
                        pass

                if os.path.exists(candidate):
                    file_path = candidate

                # Purge again after prefetching to honor the cap.
                self.__enforce_tmp_limit()

            f = open(file_path, 'rb')

            f.seek(offset)
            proto_raw = f.read(size)

            f.close()

            record = e2e_pb2.E2EDFrame()
            try:
                record.ParseFromString(proto_raw)
            except Exception as err:
                # Surface enough context to rebuild/skip the bad record and avoid silent corruption.
                raise RuntimeError(
                    f"DecodeError for file={file_path} offset={offset} size={size}: {err}"
                )

            # Build a camera-name -> image map so we can select by CameraName enum values.
            images_by_name = {img.name: img.image for img in record.frame.images}

            image_bytes = []
            missing = []
            for cam_id in self.image_ids:
                if cam_id in images_by_name:
                    image_bytes.append(images_by_name[cam_id])
                elif 0 <= cam_id < len(record.frame.images):
                    # Back-compat: allow direct index selection if provided.
                    image_bytes.append(record.frame.images[cam_id].image)
                else:
                    missing.append(cam_id)

            if missing:
                available = [img.name for img in record.frame.images]
                raise ValueError(
                    f"Requested camera ids {missing} not found. Available camera names: {available}. "
                    f"Valid index range: 0..{len(record.frame.images)-1}."
                )

            # yield images, future_states
            fs = record.future_states

            future_states = np.stack([fs.pos_x, fs.pos_y], axis=-1)   # [T, 2]

            ps = record.past_states

            past_states = np.stack([
                ps.pos_x, ps.pos_y,
                ps.vel_x, ps.vel_y,
                ps.accel_x, ps.accel_y
            ], axis=-1)  # [T, 6]

            yield {
                'past': torch.from_numpy(past_states).float(),
                'future': torch.from_numpy(future_states).float(),
                'image_bytes': image_bytes,
                'intent': record.intent
            }

            # yield torch.from_numpy(past_states), torch.from_numpy(future_states)