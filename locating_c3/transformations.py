#transformations
#created: 09/07/21
#last updated: 19/07/2021
#hermione 
#%%
from SimpleITK.extra import Resample
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
from scipy.ndimage import binary_fill_holes
from skimage.transform import rescale, resize

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

"""
def cropping(inp: np.ndarray, tar: np.ndarray ):
    #want to crop all CT scans to be [177,260,260]
    x, y= inp, tar
    print("ct max: ", np.max(x))
    _,threshold = cv2.threshold(x,500,0,cv2.THRESH_TOZERO)
    coords = center_of_mass(threshold)
    print("coords: ", coords)
    size =130
    #x, y direction cropping
    x_min = int(((coords[1] - size)+126)/2)
    x_max = int(((coords[1] + size)+386)/2)
    y_min = int(((coords[2] - size)+126)/2)
    y_max = int(((coords[2] + size)+386)/2)
    if (x.shape[0]>=117):
        print("True", x.shape[0])
    else:
        print("too small ffs")
    
    #z axis cropping
    inds = x < -500
    im = x
    im[...] = 1
    im[inds] = 0
    im = binary_fill_holes(im).astype(int)
    filled_inds = np.nonzero(im)
    print("im shape: ", im.shape, np.unique(im))
    for z in range(len(im)-1,0, -1):
        seg_slice = im[z,...]
        val = np.sum(seg_slice)
        if val != 0:
            high_cc_cut = z
            print(z)
            break
    high_cc_cut = filled_inds[0][-1]
    im = x[(high_cc_cut-116):high_cc_cut,:,:]
    #y = y[(high_cc_cut-116):high_cc_cut,:,:]

    # Cuts complete
    cutdown_shape = np.array(im.shape)
    print("cropped shape: ", cutdown_shape)
    x, y = x[(high_cc_cut-116):high_cc_cut,x_min:x_max,y_min:y_max], tar[(high_cc_cut-116):high_cc_cut,x_min:x_max,y_min:y_max]
   
    #x, y = im[(im.shape[0]-117):,x_min:x_max,y_min:y_max], tar[(im.shape[0]-117):,x_min:x_max,y_min:y_max]
    return x, y"""

def cropping(inp: np.ndarray, tar: np.ndarray ):
    #working one but z axis crop needs improving
    x, y= inp, tar
    _,threshold = cv2.threshold(x,200,0,cv2.THRESH_TOZERO)
    coords = center_of_mass(x)
    size =126
    x_min = int(((coords[1] - size)+126)/2)
    x_max = int(((coords[1] + size)+386)/2)
    y_min = int(((coords[2] - size)+126)/2)
    y_max = int(((coords[2] + size)+386)/2)
    #z crop
    z_size = 112
    z_coords = {"z_min":x.shape[0]-z_size,"z_max":x.shape[0]}
    
    if (z_size < x.shape[0] < 200):
        print("True", x.shape[0])
        #z_coords = {"z_min":x.shape[0]-z_size,"z_max":x.shape[0]}
    elif (200 < x.shape[0]):
        print("big boi", x.shape[0])
        if(x.shape[0] > int(coords[0])+z_size):
            z_coords = {"z_min": int(coords[0]), "z_max": int(coords[0])+z_size}
        
    else:
        print("too small ffs: ", x.shape[0])

    x, y = inp[z_coords["z_min"]:z_coords["z_max"],x_min:x_max,y_min:y_max], tar[(x.shape[0]-z_size):,x_min:x_max,y_min:y_max]
    
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
    skip = False
    if (max(np.unique(sphere)) != 1):
        print("problematic")
        skip = True
    #Plot the result
    # fig =plt.figure(figsize=(6,6))
    # ax = plt.axes(projection='3d')
    # ax.voxels(sphere, edgecolor='red')
    # plt.show()
    return sphere, skip

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
        x = sitk.ReadImage(input_ID, imageIO="NiftiImageIO")
        y = sitk.ReadImage(target_ID, imageIO="NiftiImageIO")
        x, y = sitk.GetArrayFromImage(x).astype(float), sitk.GetArrayFromImage(y).astype(float)
        print("max, min: ",np.max(x), np.min(x))
        print("type:", x.dtype, y.dtype)
        #voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
        
        # Preprocessing
        if self.cropping is not None:
            x, y = self.cropping(x, y)
            
        if self.heatmap is not None:
            y,_ = self.heatmap(y)

        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)
            
        if self.normalise is not None:
            x, y = self.normalise(x), self.normalise(y)
            
        #downsampling #[32,128,128]
        x = rescale(x, scale=0.5, order=0, multichannel=False, preserve_range=True, anti_aliasing=False)
        y = rescale(y, scale=0.5, order=0, multichannel=False, preserve_range=True, anti_aliasing=False)
        #x = Resample(x,[32,128,128])
        #y = Resample(y,[32,128,128])
        data = {'input': x, 'mask': y}  
        return data

def path_list(no_patients, skip: list):
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
no_patients = 35
skip = [24,25,37]#[8,11,17]
PathList =  path_list(no_patients, skip)
inputs = PathList[0]
targets = PathList[1]
ids = PathList[2]

new_patient_no = no_patients - len(skip)
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

CTs, masks = np.array(CTs), np.array(masks)

def PrintSlice(input, targets, number):
    slice_no = GetSliceNumber(targets)
    print("patient ", number)
    plt.imshow(input[slice_no,...], cmap = "gray")
    #for i in range(len(targets)):
        #targets[i,...,0][targets[i,...,0] == 0] = np.nan
    plt.imshow(targets[slice_no,...], cmap = "cool", alpha = 0.5)
    plt.axis('off')
    plt.show()

for i in range(0,no_patients):
    PrintSlice(CTs[i], masks[i], i)

#%%
#save the preprocessed masks and cts for the dataset
save_preprocessed(CTs, masks, ids)