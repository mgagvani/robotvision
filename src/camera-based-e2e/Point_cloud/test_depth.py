import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor
try:
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    print(processor)
except Exception as e:
    print(e)
