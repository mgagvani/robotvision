import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

images = [
    np.zeros((1280, 1920, 3), dtype=np.uint8), # front
    np.zeros((886, 1920, 3), dtype=np.uint8)  # side
]

original_sizes = [(img.shape[0], img.shape[1]) for img in images]
max_h = max(s[0] for s in original_sizes)
max_w = max(s[1] for s in original_sizes)

padded_images = []
for img in images:
    h, w = img.shape[:2]
    padded = cv2.copyMakeBorder(
        img, 
        top=0, bottom=max_h - h, 
        left=0, right=max_w - w, 
        borderType=cv2.BORDER_REPLICATE
    )
    padded_images.append(padded)

inputs = processor(images=padded_images, return_tensors="pt")
print("Batched tensor shape:", inputs.pixel_values.shape)
