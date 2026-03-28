"""
Run SegFormer on extracted frame images and save colored segmentation maps.
Usage:
    python viz_seg.py --idx 5000
"""
import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from pathlib import Path

SEG_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

# Cityscapes colors for each class
CITYSCAPES_COLORS = [
    (128, 64,128),  # road
    (244, 35,232),  # sidewalk
    ( 70, 70, 70),  # building
    (102,102,156),  # wall
    (190,153,153),  # fence
    (153,153,153),  # pole
    (250,170, 30),  # traffic light
    (220,220,  0),  # traffic sign
    (107,142, 35),  # vegetation
    (152,251,152),  # terrain
    ( 70,130,180),  # sky
    (220, 20, 60),  # person
    (255,  0,  0),  # rider
    (  0,  0,142),  # car
    (  0,  0, 70),  # truck
    (  0, 60,100),  # bus
    (  0, 80,100),  # train
    (  0,  0,230),  # motorcycle
    (119, 11, 32),  # bicycle
]

CITYSCAPES_NAMES = [
    'road','sidewalk','building','wall','fence','pole',
    'traffic light','traffic sign','vegetation','terrain','sky',
    'person','rider','car','truck','bus','train','motorcycle','bicycle'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=5000)
    parser.add_argument("--img_dir", type=str, default="./frame_images")
    parser.add_argument("--out_dir", type=str, default="./seg_output")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SegFormer...")
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device)
    model.eval()

    for cam in range(1, 9):
        img_path = os.path.join(args.img_dir, f"frame_{args.idx:07d}_cam{cam}.jpg")
        if not os.path.exists(img_path):
            continue

        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        
        pred = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)
        class_map = pred.argmax(dim=1).squeeze().cpu().numpy()

        # Color the segmentation
        seg_colored = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_id, color in enumerate(CITYSCAPES_COLORS):
            seg_colored[class_map == cls_id] = color

        # Side by side: original + segmentation
        combined = np.concatenate([rgb[:,:,::-1], seg_colored[:,:,::-1]], axis=1)
        out_path = os.path.join(args.out_dir, f"seg_{args.idx:07d}_cam{cam}.jpg")
        cv2.imwrite(out_path, combined)
        print(f"  Saved cam{cam}: {out_path}")

if __name__ == "__main__":
    main()
