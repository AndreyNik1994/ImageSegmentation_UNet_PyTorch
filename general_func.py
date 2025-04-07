import os
import torch
from pathlib import Path
import numpy as np
import params
import torch.nn as nn

def load_model(model, model_path: str | Path | os.PathLike, weights_only: bool=True):
    model.load_state_dict(torch.load(model_path, weights_only=weights_only))
    model.eval()
    return model

def save_model(model, name: str):
    Path(params.MODEL_SAVE_PATH).mkdir(exist_ok=True)
    save_path = params.MODEL_SAVE_PATH / f'{name}_detection.pth'

    torch.save(model.state_dict(), save_path)

def convert_labels_to_rgb(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for color_info in sorted(params.COLORS_CONFIG, key=lambda x: x['class'], reverse=True):
        class_id = color_info['class']
        rgb = color_info['rgb']
        mask = (label == class_id)
        color_mask[mask] = rgb
    return color_mask

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1.
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# criterion = DiceLoss()