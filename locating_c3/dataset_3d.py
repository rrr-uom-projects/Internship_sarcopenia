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
        # Load input and target
        #x = self.inputs[index]
        #y = self.targets[index]
        x = self.availableInputs[index]
        y = self.availableTargets[index]
        print("shape: ",x.shape)
        print("type:", x.dtype, y.dtype)
        
        def voxeldim():
            voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
            return voxel_dim
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        # Preprocessing
        if self.transform is not None:
            augs = self.transform(x,y, data_keys=["input","input"])
            x = augs[0].unsqueeze(0)
            y = augs[1].unsqueeze(0).long()
            #print("image shape: ", x.shape)
            #print("Target shape: ", y.shape)
        
        return x, y

def get_data():
    #path_O = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed.npz'
    path_H = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_8rs.npz'

    data = np.load(path_H)
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
head_augmentations = AugmentationSequential(K.RandomHorizontalFlip3D(p = 0.5),
                            K.RandomRotation3D([0, 0, 30], p = 0.5),
                            data_keys=["input" ,"input"],
                            keepdim = True,
                            )

"""
#initialise dataset

#training_dataset = Segmentation3DDataset(inputs=inputs, targets=targets, transform=augmentations)#transform=augmentations


#dataloader
#training_dataloader = DataLoader(dataset=training_dataset, batch_size=2,  shuffle = False)
#x, y = next(iter(training_dataloader))

# print(f'x = shape: {x.shape}; type: {x.dtype}')
# print(f'x = min: {x.min()}; max: {x.max()}')
# print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

#x_new = x.permute(2,3,4,0,1).squeeze()
#print(x_new.shape)
#plt.imshow(x_new[83,:,:,0], cmap = "gray")
#plt.show()

# def PrintSlice(input, targets):

#     new = np.asarray((input.squeeze()).permute(1,2,3,0))
#     new_target = np.asarray((targets.squeeze()).permute(1,2,3,0))
#     slice_no = GetSliceNumber(new_target[...,0])
#     print(slice_no)
#     #slice_no=62
#     print(new_target.shape)
#     plt.imshow(new[slice_no,:,:,0], cmap = "gray")
#     #for i in range(len(new_target)):
#         #new_target[i,...,0][new_target[i,...,0] == 0] = np.nan
#     plt.imshow(new_target[slice_no,:,:,0], cmap = "cool", alpha = 0.5)
#     plt.axis('off')
#     plt.show()

# PrintSlice(x, y)

"""