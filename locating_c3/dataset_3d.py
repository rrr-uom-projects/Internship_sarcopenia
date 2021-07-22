#09/07/2021
#3D Unet to locate C3 from 3D CT scans

#https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55

#creating dataset and dataloader
#put the training images into input and target directories

from kornia.geometry import transform
#from sklearn import preprocessing
import torch
import cv2
from skimage.io import imread
#from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, Dataset
import SimpleITK as sitk
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential 
from kornia.utils import image_to_tensor, tensor_to_image
from torchvision.transforms import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import GetSliceNumber

class Segmentation3DDataset(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Load input and target
        x = self.inputs[index]
        y = self.targets[index]
        print("shape: ",x.shape)
        print("type:", x.dtype, y.dtype)
        
        def voxeldim():
            voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
            return voxel_dim
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        # Preprocessing
        if self.transform is not None:
            augs = self.transform(x, y)

            #x, y = self.transform(x), self.transform(y)
            x = augs['input']
            y = augs['mask']
        
        return x, y

def get_data():
    path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed.npz'
    data = np.load(path)
    #print([*data.keys()])
    inputs = data['inputs']
    targets = data['masks']
    ids = data['ids']
    return np.asarray(inputs), np.asarray(targets), np.asarray(ids)

#main
data = get_data()
inputs = data[0]
targets = data[1]
ids = data[2]

print("inputs: ", inputs.shape)

#augmentation
# augmentations = AugmentationSequential(K.RandomHorizontalFlip3D(p = 1),
#                             K.RandomRotation3D([0, 0, 30], p = 1),
#                             data_keys=["input","mask"],
#                             )

#initialise dataset
training_dataset = Segmentation3DDataset(inputs=inputs, targets=targets)#transform=augmentations

#dataloader
training_dataloader = DataLoader(dataset=training_dataset, batch_size=2,  shuffle = False)
x, y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

#x_new = x.permute(2,3,4,0,1).squeeze()
#print(x_new.shape)
#plt.imshow(x_new[83,:,:,0], cmap = "gray")
#plt.show()

# def PrintSlice(input, targets):
#     new = np.asarray(input.permute(3,4,5,0,1,2).squeeze())
#     #slice_no = GetSliceNumber(targets[0])
#     slice_no=62
#     new_target = np.asarray(targets.permute(3,4,5,0,1,2).squeeze())
#     print(new_target.shape)
#     plt.imshow(new[slice_no,:,:,0], cmap = "gray")
#     #for i in range(len(new_target)):
#         #new_target[i,...,0][new_target[i,...,0] == 0] = np.nan
#     plt.imshow(new_target[slice_no,:,:,0], cmap = "cool", alpha = 0.5)
#     plt.axis('off')
#     plt.show()

# PrintSlice(x, y)