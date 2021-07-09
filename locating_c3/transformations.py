#transformations
#09/07/21

#hermione

import numpy as np
import albumentations as A
import random
from albumentations.pytorch import ToTensor

def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (clipping)"""
    window = 350
    level = 50
    vmax = level/2 + window
    vmin = level/2-window
    inp[inp > vmax] = vmax
    inp[inp<vmin] = vmin
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def normalize(inp: np.ndarray, mean: float, std: float):
    """Normalize based on mean and standard deviation."""
    inp_out = (inp - mean) / std
    return inp_out
