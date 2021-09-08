#transformations
#created: 09/07/21
#last updated: 09/08/2021
#hermione 
#%%
from kornia.augmentation.augmentation3d import CenterCrop3D
import numpy as np
import SimpleITK as sitk
import random
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter
import torch
from utils import GetSliceNumber, GetTargetCoords, projections, mrofsnart, display_input_data, slice_preds, do_it_urself_round
import cv2
import os
import matplotlib.pyplot as plt
#from scipy.ndimage import binary_fill_holes
from skimage.transform import rescale
import sklearn
from sklearn import preprocessing

# import tarfile
# path = '/data/sarcopenia/HnN/c3_location.tar'
# tar = tarfile.open(path)
# tar.extractall()
# tar.close()
#%%
def window_level(inp: np.array):
    window = 350
    level = 50
    vmax = level/2 + window
    vmin = level/2-window
    thresh = 1000
    inp[inp > thresh] = 0
    inp[inp > vmax] = vmax
    inp[inp < vmin] = vmin
    #print("wl: ", inp.shape, np.max(inp), np.min(inp))
    return inp

def windowLevelNormalize(image): #eds
    window = 350
    level = 50
    minval = level - window/2
    maxval = level + window/2
    thresh = 1000
    image[image > thresh] = 0
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld /= window
    return wld

def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (plus clipping)"""
    inp = (np.nan_to_num(inp)).astype(np.float64)
    shape = inp.shape
    inp_new = np.round(sklearn.preprocessing.minmax_scale(inp.ravel(), feature_range=(0,1)), decimals = 10).reshape(shape)
    #print("norm: ", inp_new.shape, np.max(inp_new), np.min(inp_new))
    #inp_out = (inp - np.min(inp)) / np.ptp(inp)
    #inp_out =(inp-np.min(inp)) / np.max(inp)
    inp_out = inp_new
    if np.min(inp_out) < 0:
        print("problem here\n")
        inp_out[inp_out < 0] = 0
    if np.max(inp_out) == 0:
        print("another problem here\n")
        print("HELP\n")
    #print("norm: ", np.max(inp_out), np.min(inp_out))
    return inp_out

def normalize(inp: np.ndarray, mean: float, std: float):
    """Normalize based on mean and standard deviation."""
    inp_out = (inp - mean) / std
    return inp_out


def cropping(inp: np.ndarray, tar: np.ndarray ):
    #working one but z axis crop needs improving
    #print("original shape: ",inp.shape)
    x, y= inp, tar
    _,threshold = cv2.threshold(x,200,0,cv2.THRESH_TOZERO)
    coords = center_of_mass(threshold)
    size =126
    x_min = do_it_urself_round(((coords[1] - size)+126)/2)
    x_max = do_it_urself_round(((coords[1] + size)+386)/2)
    y_min = do_it_urself_round(((coords[2] - size)+126)/2)
    y_max = do_it_urself_round(((coords[2] + size)+386)/2)
    #z crop
    z_size = 112
    z_coords = {"z_min":x.shape[0]-z_size,"z_max":x.shape[0]}
    org_inp_size = x.shape
        
    if (z_size < x.shape[0]):
        #print("bigger", x.shape[0])
        if(x.shape[0] > do_it_urself_round(coords[0])+z_size): #190>87+112=199
            print("crop")
            z_coords = {"z_min": do_it_urself_round(coords[0]), "z_max": do_it_urself_round(coords[0])+z_size}

    else:
        #print("too small: ", x.shape[0])
        padded_arr = np.pad(x, ((do_it_urself_round((z_size-x.shape[0])/2), do_it_urself_round((z_size-x.shape[0])/2)), (0,0),(0,0)),'mean')
        padded_tar = np.pad(y, ((do_it_urself_round((z_size-x.shape[0])/2), do_it_urself_round((z_size-x.shape[0])/2)), (0,0),(0,0)),'mean')
        inp, tar =padded_arr, padded_tar
        z_coords = {"z_min": 0, "z_max": inp.shape[0]}
    
    print(z_coords["z_min"],z_coords["z_max"],x_min)
    x, y = inp[z_coords["z_min"]:z_coords["z_max"],x_min:x_max,y_min:y_max], tar[z_coords["z_min"]:z_coords["z_max"],x_min:x_max,y_min:y_max]
    cropped_info = [z_coords["z_min"], z_coords["z_max"], org_inp_size[0], x_min, x_max, org_inp_size[1], y_min, y_max, org_inp_size[2]]
    
    return x, y, np.array(cropped_info)

def sphereMask(tar: np.ndarray):
    def create_bin_sphere(arr_size, center, r):
        coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
        distance = np.sqrt(((coords[0] - center[0])/0.5)**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
        return 1*(distance <= r)
    
    arr_size = tar.shape
    sphere_center = center_of_mass(tar)
    r=5
    sphere = create_bin_sphere(arr_size, sphere_center, r)
    #print("sphere details", sphere.shape, np.unique(sphere))
    #Plot the result
    # fig =plt.figure(figsize=(6,6))
    # ax = plt.axes(projection='3d')
    # ax.voxels(sphere, edgecolor='red')
    # plt.show()
    return sphere

def gaussian(msk):
    msk *= 100
    gauss = gaussian_filter(msk, (1,3,3) ,truncate=100)
    #print("g: ",np.max(gauss), np.min(gauss))
    #gauss = 1/(1 + np.exp(-gauss))
    return gauss

def flip(im):
    # flipped = False
    # if im.GetDirection()[-1] == -1:
    #     print("Image upside down, CC flip required!")
    #     im = sitk.GetArrayFromImage(im).astype(float)
    #     im = np.flip(im, axis=0)        # flip CC
    #     im = np.flip(im, axis=2)        # flip LR --> this works, should be made more robust though (with sitk cosine matrix)
    #     #im = im[::-1, :, :]
    #     flipped = True   
    #     #print("flipped") 
    # else:
    #     im = sitk.GetArrayFromImage(im).astype(float)

    im = np.flip(im, axis=0)        # flip CC
    im = np.flip(im, axis=2)        # flip LR
    return im

def path_list(no_patients, skip: list):
    path_list_inputs = []
    path_list_targets = []
    ids = []

    for i in range(1,no_patients+1):
        if i not in skip:
            path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/'
            #path = 'C:/Users/hermi/OneDrive/Documents/physics year 4/Mphys/L3_scans/My_segs'
            path_list_inputs.append(path + "/P" + str(i) + "_RT_sim_ct.nii.gz")
            path_list_targets.append(path + "/P" + str(i) + "_RT_sim_seg.nii.gz")
            id = "01-00" + str(i)
            ids.append(id)
    print("no read in: ", len(path_list_inputs))

    return np.array(path_list_inputs), np.array(path_list_targets), np.array(ids)

def getFiles(targetdir):
    ls = []
    ids = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue    # skip directories
        ls.append(path)
        ids.append(fname)
    return ls, ids

def path_list2():
    im_dir = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1/images'
    msk_dir = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1/masks'
    inputs = getFiles(im_dir)
    path_list_inputs = inputs[0]
    path_list_targets = getFiles(msk_dir)[0]
    ids = inputs[1]
    return path_list_inputs, path_list_targets, ids

def save_preprocessed(inputs, targets, ids, org_slice_nos, voxel_dims, transforms = None):
    path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_TFinal2_gauss.npz'
    vox_path =  '/home/hermione/Documents/Internship_sarcopenia/locating_c3/vox_dims.npz'
    #path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed_Tgauss.npz' 
    ids = np.array(ids)
    print("final shape: ", inputs.shape, targets.shape, ids.shape, len(org_slice_nos), len(voxel_dims))
    np.savez(path, inputs = inputs.astype(np.float32), masks = targets.astype(np.float32), ids = ids, transforms = transforms, org_nos = org_slice_nos.astype(int), dims = voxel_dims.astype(np.float32))
    #sep file for voxels
    np.savez(vox_path, ids = ids, dims = voxel_dims.astype(np.float32))
    print("Saved preprocessed data") 

class preprocessing():
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform = window_level, cropping = None, normalise = None, 
                 heatmap = None, sphere = None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.cropping = cropping
        self.normalise = normalise
        self.heatmap = heatmap
        self.sphere  = sphere
        self.transform_list = []
        self.slices_gt = []
        self.voxel_dims_list = []

    def __len__(self):
        return len(self.inputs)

    def transforms(self):
        #maybe save it here
        return np.array(self.transform_list)
    
    def original_slices(self):
        return np.array(self.slices_gt)
    
    def voxel_dims(self):
        return np.array(self.voxel_dims_list)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]
        #transform_list_item = []
        # Load input and target
        x = sitk.ReadImage(input_ID, imageIO="NiftiImageIO")
        y = sitk.ReadImage(target_ID, imageIO="NiftiImageIO")
        # check if flip required
        need_flip = False
        if x.GetDirection()[-1] == -1:
            need_flip = True

        #saving the spacing
        voxel_dim = [(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
        self.voxel_dims_list.append(np.array(voxel_dim).astype(np.float32))

        x, y = sitk.GetArrayFromImage(x).astype(float), sitk.GetArrayFromImage(y).astype(float)
        x-=1024

        #save original slice number
        slice_no = GetTargetCoords(y)[0]
        #print("Original Slice No: ", GetTargetCoords(y))
        self.slices_gt.append(slice_no)

        # Preprocessing
        if need_flip == True:
            x,y = flip(x), flip(y)

        #print("Post flip: ", GetTargetCoords(y))

        if self.transform is not None:
            x = self.transform(x)
    
        if self.cropping is not None:
            x, y, crop_info = self.cropping(x, y)
       # print("Post cropping: ", GetTargetCoords(y))
        if self.sphere is not None:
            y = self.sphere(y)

        if self.heatmap is not None: #and self.sphere is None
            y = self.heatmap(y)

        if self.normalise is not None:
            x, y = self.normalise(x), self.normalise(y)

        #save transforms and gt slice
        transform_list_item = [need_flip, crop_info]
        self.transform_list.append(np.array(transform_list_item))
        #downsampling to size -> [128,128,128]
        x = rescale(x, scale=((16/14),0.5,0.5), order=0, multichannel=False,  anti_aliasing=False)
        y = rescale(y, scale=((16/14),0.5,0.5), order=0, multichannel=False,  anti_aliasing=False)
        #print("Post scale: ", GetTargetCoords(y))
        assert np.min(y) >= 0
        assert np.max(y) > 0
        data = {'input': x, 'mask': y}  
        return data


#main
#get the file names
PathList =  path_list2()
no_patients = 2
inputs = PathList[0]#[:no_patients]
targets = PathList[1]#[:no_patients]
ids = PathList[2]#[:no_patients]

print("no of patients: ",len(inputs))
#apply preprocessing
preprocessed_data = preprocessing(inputs=inputs, targets=targets, normalise = normalize_01, cropping = cropping, heatmap= gaussian)#sphere = sphereMask

CTs = []
masks = []
for i in range(len(preprocessed_data)):
    sample = preprocessed_data[i]
    print(i, "inp: ", sample['input'].shape, "msk: ", sample['mask'].shape)
    x, y = sample['input'], sample['mask']
    CTs.append(x)
    masks.append(y)


CTs, masks = np.array(CTs), np.array(masks)   

transforms = preprocessed_data.transforms()
org_slices = preprocessed_data.original_slices()
voxel_dims = preprocessed_data.voxel_dims()
print(org_slices)

#final_transformed_slices = slice_preds(masks)
#x, y, z = mrofsnart(masks, transforms)
#print(z)
#%%
#save the preprocessed masks and cts for the dataset
save_preprocessed(CTs, masks, ids, org_slices, voxel_dims, transforms)

#%%
#path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_Tgauss.npz' 
#display_input_data(path)

