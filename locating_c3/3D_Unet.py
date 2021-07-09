#09/07/2021
#3D Unet to locate C3 from 3D CT scans

#https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55

#creating dataset and dataloader
#put the training images into input and target directories

from sklearn import preprocessing
import torch
from skimage.io import imread
#from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, Dataset
import SimpleITK as sitk
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

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
        self.targets_dtype = torch.long

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
        #cropping so they are the same size [512,512,117,1]
        x, y = x[:117,...], y[:117,...]
        print("shape: ",x.shape)
        print("type:", x.dtype, y.dtype)
        
        def voxeldim():
            voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
            return voxel_dim

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y



def path_list(no_patients, skip = []):
    path_list_inputs = []
    path_list_targets = []
    ids = []
    for i in range(1,no_patients+1):
        if i not in skip:
            path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/'
            path_list_inputs.append(path + "inputs/P" + str(i) + "_RT_sim_ct.nii.gz")
            path_list_targets.append(path + "targets/P" + str(i) + "_RT_sim_seg.nii.gz")
            id = "01-00" + str(i)
            ids.append(id)
    return path_list_inputs, path_list_targets, ids

#main
inputs = np.array(path_list(3)[0])
targets = np.array(path_list(3)[1])

print(inputs.shape)
"""
def ReadIn(input_ID, target_ID):
    x = sitk.ReadImage(input_ID, imageIO="NiftiImageIO")
    y = sitk.ReadImage(target_ID, imageIO="NiftiImageIO")
    voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
    
    x, y = sitk.GetArrayFromImage(x).astype(float), sitk.GetArrayFromImage(y).astype(float)
    #cropping so they are the same size [512,512,117,1]
    #x, y = x[:,:117,...], y[:,:117,...]
    print("shape: ",x.shape)
    print("type:", x.dtype, y.dtype)
            
    return x,y,voxel_dim

for i in range(0, len(inputs)):
    x,y = [],[]
    print(i)
    x.append(ReadIn(inputs[i], targets[i])[0])
    y.append(ReadIn(inputs[i], targets[i])[1])
    print(x.shape)
"""

training_dataset = Segmentation3DDataset(inputs=inputs, targets=targets, transform=None)


training_dataloader = DataLoader(dataset=training_dataset, batch_size=2,  shuffle=True)
x, y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')