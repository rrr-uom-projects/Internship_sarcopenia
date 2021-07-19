#transformations
#created: 09/07/21
#last updated: 19/07/2021
#hermione 

import numpy as np
import SimpleITK as sitk
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
                 transform=None, cropping = None, normalise = None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.cropping = cropping
        self.normalise = normalise

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        #x, y = imread(input_ID), imread(target_ID)
        x = sitk.ReadImage(input_ID, imageIO="NiftiImageIO")
        y = sitk.ReadImage(target_ID, imageIO="NiftiImageIO")
        x, y = sitk.GetArrayFromImage(x).astype(float), sitk.GetArrayFromImage(y).astype(float)
        #cropping so they are the same size [512,512,117,1] #but do this before
        x, y = x[:117,...], y[:117,...]
        print("shape: ",x.shape)
        print("type:", x.dtype, y.dtype)
        
        # def voxeldim(): #save this to file
        #     voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
        #     return voxel_dim

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)
        
        if self.cropping is not None:
            x, y = self.cropping(x), self.cropping(y)

        if self.normalise is not None:
            x, y = self.normalise(x), self.normalise(y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y