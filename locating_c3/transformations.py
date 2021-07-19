#transformations
#created: 09/07/21
#last updated: 19/07/2021
#hermione 

import numpy as np
import albumentations as A
import random
from albumentations.pytorch import ToTensor
from sklearn import preprocessing
import torch

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

def cropping(x,y):
    x, y = x[:117,...], y[:117,...]
    return x, y

class preprocessing():
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]