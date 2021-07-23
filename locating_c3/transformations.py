#transformations
#created: 09/07/21
#last updated: 19/07/2021
#hermione 

from kornia.augmentation.augmentation3d import CenterCrop3D
import numpy as np
import SimpleITK as sitk
#import albumentations as A
import random
#from albumentations.pytorch import ToTensor
from scipy.ndimage.measurements import center_of_mass
from sklearn import preprocessing
import torch
from utils import GetSliceNumber, Guassian
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage import measure

def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (plus clipping)"""
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

def cropping(inp: np.ndarray, tar: np.ndarray ):
    x, y= inp, tar
    print("ct max: ", np.max(x))
    _,threshold = cv2.threshold(x,200,0,cv2.THRESH_TOZERO)
    coords = center_of_mass(x)
    print("coords: ", coords)
    size =130
    x_min = int(((coords[1] - size)+126)/2)
    x_max = int(((coords[1] + size)+386)/2)
    y_min = int(((coords[2] - size)+126)/2)
    y_max = int(((coords[2] + size)+386)/2)
    x, y = inp[:117,x_min:x_max,y_min:y_max], tar[:117,x_min:x_max,y_min:y_max]
    return x, y

def sphereMask(tar: np.ndarray):
    def create_bin_sphere(arr_size, center, r):
        coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
        distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
        return 1*(distance <= r)
    
    arr_size = tar.shape
    sphere_center = center_of_mass(tar)
    r=3
    sphere = create_bin_sphere(arr_size, sphere_center, r)
    print("sphere details", sphere.shape, np.unique(sphere))
    #Plot the result
    # fig =plt.figure(figsize=(6,6))
    # ax = plt.axes(projection='3d')
    # ax.voxels(sphere, edgecolor='red')
    # plt.show()
    return sphere

class preprocessing():
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None, cropping = None, normalise = None, heatmap = None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.cropping = cropping
        self.normalise = normalise
        self.heatmap = heatmap

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
        print("max: ",np.max(x))
        print("type:", x.dtype, y.dtype)
        #voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
        
        # Preprocessing
        if self.cropping is not None:
            x, y = self.cropping(x, y)
            #data = self.cropping(data)
        
        if self.heatmap is not None:
            y = self.heatmap(y)
            print("adding sphere: ", y.shape)

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
            #path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/'
            path = 'C:/Users/hermi/OneDrive/Documents/physics year 4/Mphys/L3_scans/My_segs'
            # path_list_inputs.append(path + "inputs/P" + str(i) + "_RT_sim_ct.nii.gz")
            # path_list_targets.append(path + "targets/P" + str(i) + "_RT_sim_seg.nii.gz")
            path_list_inputs.append(path + "/P" + str(i) + "_RT_sim_ct.nii.gz")
            path_list_targets.append(path + "/P" + str(i) + "_RT_sim_seg.nii.gz")
            id = "01-00" + str(i)
            ids.append(id)
    print("no read in: ", len(path_list_inputs))
    return np.array(path_list_inputs), np.array(path_list_targets), np.array(ids)

def save_preprocessed(inputs, targets, ids):
    #path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed.npz'    
    path = 'C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\Mphys sem 2\\summer internship\\Internship_sarcopenia\\locating_c3\\preprocessed.npz'
    print("final shape: ", inputs.shape, targets.shape, ids.shape)
    for i in range(len(targets)):
        print("slice no: ",GetSliceNumber(targets[i]))
    np.savez(path, inputs = inputs, masks = targets, ids = ids)
    print("Saved preprocessed data")


#main
#get the file names
no_patients = 3
PathList =  path_list(no_patients)
inputs = PathList[0]
targets = PathList[1]
ids = PathList[2]

print(inputs.shape)
#print(targets.shape)

#apply preprocessing
preprocessed_data = preprocessing(inputs=inputs, targets=targets, normalise = normalize_01, cropping = cropping, heatmap= sphereMask)

CTs = []
masks = []
for i in range(len(preprocessed_data)):
    sample = preprocessed_data[i]
    print(i, "inp: ", sample['input'].shape, "msk: ", sample['mask'].shape)
    x, y = sample['input'], sample['mask']
    CTs.append(x)
    masks.append(y)


# for i in range(len(preprocessed_data)):
#     samples = next(iter(preprocessed_data))
#     x, y = samples['input'], samples['mask']
#     CTs.append(x)
#     masks.append(y)
#     print(i)
#     print(f'x = shape: {x.shape}; type: {y.dtype}')
#     print(f'x = min: {x.min()}; max: {x.max()}')
#     print(f'y = shape: {y.shape}; class: {np.unique(y)}; type: {y.dtype}')
CTs, masks = np.array(CTs), np.array(masks)

#save the preprocessed masks and cts for the dataset
save_preprocessed(CTs, masks, ids)

def PrintSlice(input, targets):
    slice_no = GetSliceNumber(targets)
    plt.imshow(input[slice_no,...], cmap = "gray")
    #for i in range(len(targets)):
        #targets[i,...,0][targets[i,...,0] == 0] = np.nan
    plt.imshow(targets[slice_no,...], cmap = "cool", alpha = 0.5)
    plt.axis('off')
    plt.show()

PrintSlice(CTs[2], masks[2])