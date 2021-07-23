
print("test")

from kornia.geometry import transform
#from sklearn import preprocessing
import torch
from skimage.io import imread
#from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, Dataset
import SimpleITK as sitk
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
import kornia.augmentation as K
#import kornia.augmentation.augmentation3d 
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import GetSliceNumber
import DUnet
from DUnet import UNet
from dataset_3d import Segmentation3DDataset, get_data
from trainer import Trainer
from EdHead import headHunter
#retrieving data


# device
if torch.cuda.is_available():
    device = torch.device('cuda:2')
else:
    torch.device('cpu')
#Creating model

model = headHunter(filter_factor=2, targets = 1, in_channels = 3)


data = get_data()
inputs = data[0]
targets = data[1]
ids = data[2]


#augmentation
augmentations = nn.Sequential(K.RandomHorizontalFlip3D(p = 0),
                            K.RandomRotation3D([20, 0, 0],p=0))




#initialise dataset
training_dataset = Segmentation3DDataset(inputs=inputs, targets=targets)#transform=augmentations

#dataloader
training_dataloader = DataLoader(dataset=training_dataset, batch_size=2,  shuffle=True)
