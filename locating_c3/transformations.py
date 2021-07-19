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
    for i in range(len(inp)):
        inp[i][inp[i] > vmax] = vmax
        inp[i][inp[i] < vmin] = vmin
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def normalize(inp: np.ndarray, mean: float, std: float):
    """Normalize based on mean and standard deviation."""
    inp_out = (inp - mean) / std
    return inp_out

def cropping(inp: np.ndarray):
    x = inp[:117,...]
    return x,

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
        if self.cropping is not None:
            x, y = self.cropping(x), self.cropping(y)
            #data = self.cropping(data)

        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)
            #data = self.transform(data)

        if self.normalise is not None:
            x, y = self.normalise(x), self.normalise(y)
            #data = self.normalise(data)

        # Typecasting
        #x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        data = {'input': x, 'mask': y}  
        return data

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
    return np.array(path_list_inputs), np.array(path_list_targets), np.array(ids)

def save_preprocessed(inputs, targets, ids):
    path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed.npz'    
    np.savez(path, inputs = inputs, masks = targets, ids = ids)
    print("Saved preprocessed data")


#main
#get the file names
no_patients = 3
inputs = path_list(no_patients)[0]
targets = path_list(no_patients)[1]
ids = path_list(no_patients)[2]

print(inputs.shape)

#apply preprocessing
preprocessed_data = preprocessing(inputs=inputs, targets=targets, normalise = normalize_01, cropping = cropping)

for i in range(len(preprocessed_data)):
    sample = preprocessed_data[i]
    print(i, "inp: ", sample['input'].shape, "msk: ", sample['mask'].shape)

CTs = []
masks = []
for i in range(len(preprocessed_data)):
    samples = next(iter(preprocessed_data))
    x, y = samples['input'], samples['mask']
    CTs.append(x)
    masks.append(y)
    print(i)
    print(f'x = shape: {x.shape}; type: {y.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {np.unique(y)}; type: {y.dtype}')
CTs, masks = np.array(CTs), np.array(masks)

#save the preprocessed masks and cts for the dataset
save_preprocessed(CTs, masks, ids)