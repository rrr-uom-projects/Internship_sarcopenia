# 09/07/2021
# Hermione Warr and Olivia Murray
# Dataset to read in, split, preprocess and return data for Neck Navigator model

#imports
from kornia.geometry import transform
import torch
import cv2
from skimage.io import imread
from torch.utils.data import DataLoader, TensorDataset, Dataset
import SimpleITK as sitk
import numpy as np
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential 
from kornia.utils import image_to_tensor, tensor_to_image
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import GetSliceNumber, projections


# Neck Navigator Dataset
class neckNavigatorDataset(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 image_inds:list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.availableInputs = [inputs[ind] for ind in image_inds]
        self.availableTargets = [targets[ind] for ind in image_inds]

    def __len__(self):
        return len(self.availableInputs)

    def __getitem__(self, index: int):

        x = self.availableInputs[index]
        y = self.availableTargets[index]

        # Calculating voxel spacing
        def voxeldim():
            voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
            return voxel_dim
        
        #Typecasting
        #print(np.unique(y), y.dtype )
        x = (torch.from_numpy(x)).type(self.inputs_dtype)
        y = (torch.from_numpy(y)).type(self.targets_dtype)
        #print("tensor type", y.dtype)
        #print(torch.unique(y))
        # Preprocessing
        if self.transform is not None:
            y = K.RandomAffine3D(0,[0,1,1],p=0.5,keepdim = True)(y)#random mask shifts side to side
            augs = self.transform(x,y, data_keys=["input","input"])
            x = augs[0]
            y = augs[1] 
     
        # creating channel dimension            
        return x.unsqueeze(0), y.unsqueeze(0)


# functions and definitions
def get_data(path): 
    data = np.load(path)
    inputs = data['inputs']
    targets = data['masks']
    ids = data['ids']
    return np.asarray(inputs), np.asarray(targets), np.asarray(ids)


#augmentation
head_augmentations = AugmentationSequential(K.RandomHorizontalFlip3D(p = 0.5),
                            K.RandomRotation3D([0, 5, 20], p = 0.5),
                            data_keys=["input" ,"input"],
                            keepdim = True,
                            )

def mask_aug(msk, p):
    #want an aug to shift masks on x and y axis randomly.
    shift = np.random.choice(np.arange(-6, 5), p=[p/10,p/10,p/10,p/10,p/10,1-p,p/10,p/10,p/10,p/10,p/10])
    shift_mask = np.roll(msk, (shift,shift),axis=(1,2))
    plt.imshow(msk, cmap = 'cool')
    plt.axis('off')
    plt.show()
    return shift_mask
