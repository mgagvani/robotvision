import json
import torch
import torch.nn as nn
import torch.nn.Functional as F

losses = json.load(open("scene_loss.json"))

def generate_weights(loss_dict, epsilon=0.001, normalize=True):
    items = list(loss_dict.items())
    items.sort(key=lambda x: (-x[1], x[0]))
    n = len(items)
    weights = {}
    for rank, (frame_id, loss) in enumerate(items, start=1):
        w = (n - rank + 1) + epsilon
        weights[frame_id] = float(w)
    if normalize:
        s = sum(weights.values())
        weights = {k: v/s for k, v in weights.items()}
    return weights

def sample_frame_ids(weight_by_frame, k=1):
    frame_ids = list(weight_by_frame.keys())
    w = list(weight_by_frame.values())
    return random.choices(frame_ids, weights=w, k=k)