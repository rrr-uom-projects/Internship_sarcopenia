
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

#retrieving data



#Creating model
model = UNet(in_channels = 1,
 out_channels = 2,
 n_blocks = 3,
 start_filts= 32,
 up_mode ='transpose',
 merge_mode = 'concat',
 planar_blocks = (),
 batch_norm='unset',
 attention= False,
 activation = 'relu',
 normalization= 'batch',
 full_norm= True,
 dim=3,
 conv_mode= 'same')

def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result



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
x, y = next(iter(training_dataloader))


