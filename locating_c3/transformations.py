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
from utils import GetSliceNumber, Guassian, projections, PrintSlice
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

def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (plus clipping)"""
    inp = (np.nan_to_num(inp)).astype(np.float64)
    shape = inp.shape
    inp_new = np.round(sklearn.preprocessing.minmax_scale(inp.ravel(), feature_range=(0,1)), decimals = 10).reshape(shape)
    print("norm: ", inp_new.shape, np.max(inp_new), np.min(inp_new))
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
    x, y= inp, tar
    _,threshold = cv2.threshold(x,200,0,cv2.THRESH_TOZERO)
    coords = center_of_mass(threshold)
    size =126
    x_min = int(((coords[1] - size)+126)/2)
    x_max = int(((coords[1] + size)+386)/2)
    y_min = int(((coords[2] - size)+126)/2)
    y_max = int(((coords[2] + size)+386)/2)
    #z crop
    z_size = 112
    z_coords = {"z_min":x.shape[0]-z_size,"z_max":x.shape[0]}
    #z_coords = {"z_min":0,"z_max":z_size}
    
    if (z_size < x.shape[0] < z_size):
        print("True", x.shape[0])
        
    elif (z_size < x.shape[0]):
        print("bigger", x.shape[0])
        if(x.shape[0] > int(coords[0])+z_size): #190>87+112=199
            print("crop")
            z_coords = {"z_min": int(coords[0]), "z_max": int(coords[0])+z_size}

    else:
        print("too small: ", x.shape[0])
        padded_arr = np.pad(x, ((int((z_size-x.shape[0])/2),int((z_size-x.shape[0])/2)), (0,0),(0,0)),'mean')
        padded_tar = np.pad(y, ((int((z_size-x.shape[0])/2),int((z_size-x.shape[0])/2)), (0,0),(0,0)),'mean')
        inp, tar =padded_arr, padded_tar
        z_coords = {"z_min": 0, "z_max": inp.shape[0]}
    
    #print("y pre chopped: ", np.max(y), np.min(y))   
    #print("z_coords: ",z_coords["z_min"], z_coords["z_max"] )
    x, y = inp[z_coords["z_min"]:z_coords["z_max"],x_min:x_max,y_min:y_max], tar[z_coords["z_min"]:z_coords["z_max"],x_min:x_max,y_min:y_max]
    #print(x.shape, y.shape)
    #print("y chopped: ", np.max(y), np.min(y))
    return x, y

def sphereMask(tar: np.ndarray):
    def create_bin_sphere(arr_size, center, r):
        coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
        distance = np.sqrt(((coords[0] - center[0])/0.5)**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
        return 1*(distance <= r)
    
    arr_size = tar.shape
    sphere_center = center_of_mass(tar)
    r=5
    sphere = create_bin_sphere(arr_size, sphere_center, r)
    print("sphere details", sphere.shape, np.unique(sphere))
    #Plot the result
    # fig =plt.figure(figsize=(6,6))
    # ax = plt.axes(projection='3d')
    # ax.voxels(sphere, edgecolor='red')
    # plt.show()
    return sphere

def gaussian(msk):
    msk *= 100
    gauss = gaussian_filter(msk, (1,3,3) ,truncate=100)
    print("g: ",np.max(gauss), np.min(gauss))
    #gauss = 1/(1 + np.exp(-gauss))
    #print("g sig: ",np.max(gauss), np.min(gauss), np.unique(gauss))
    return gauss

def flip(im):
    flipped = False
    if im.GetDirection()[-1] == -1:
        print("Image upside down, CC flip required!")
        im = sitk.GetArrayFromImage(im).astype(float)
        im = np.flip(im, axis=0)        # flip CC
        im = np.flip(im, axis=2)        # flip LR --> this works, should be made more robust though (with sitk cosine matrix)
        #im = im[::-1, :, :]
        flipped = True   
        #print("flipped") 
    else:
        im = sitk.GetArrayFromImage(im).astype(float)
    return im

def path_list(no_patients, skip: list):
    path_list_inputs = []
    path_list_targets = []
    ids = []

    for i in range(1,no_patients+1):
        if i not in skip:
            #path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/'
            path = 'C:/Users/hermi/OneDrive/Documents/physics year 4/Mphys/L3_scans/My_segs'
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

def save_preprocessed(inputs, targets, ids):
    path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_guass2.npz' 
    ids = np.array(ids)   
    #path = 'C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\Mphys sem 2\\summer internship\\Internship_sarcopenia\\locating_c3\\preprocessed.npz'
    print("final shape: ", inputs.shape, targets.shape, ids.shape)
    np.savez(path, inputs = inputs.astype(np.float32), masks = targets.astype(np.float32), ids = ids)
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
        #self.inputs_dtype = torch.float32
        #self.targets_dtype = torch.long
        self.cropping = cropping
        self.normalise = normalise
        self.heatmap = heatmap
        self.sphere  =sphere

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
        # check if flip required
        x,y = flip(x), flip(y)
        #x, y = sitk.GetArrayFromImage(x).astype(float), sitk.GetArrayFromImage(y).astype(float)
        #x,y = flip(x), flip(y)
        x-=1024
        #voxel_dim = np.array[(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
        #print("y start: ", np.max(y), np.min(y))
        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)
    
        if self.cropping is not None:
            x, y = self.cropping(x, y)

        if self.sphere is not None:
            y = self.sphere(y)

        if self.heatmap is not None: #and self.sphere is None
            y = self.heatmap(y)

        if self.normalise is not None:
            x, y = self.normalise(x), self.normalise(y)

        #downsampling #[32,128,128]
        x = rescale(x, scale=((16/14),0.5,0.5), order=0, multichannel=False,  anti_aliasing=False)
        y = rescale(y, scale=((16/14),0.5,0.5), order=0, multichannel=False,  anti_aliasing=False)
       
        #print("shape: ", x.shape, y.shape)
        #print("max, min: ", np.max(x), np.min(x))
        print("y being a little shit: ", np.max(y), np.min(y))
        assert np.min(y) >= 0
        #assert np.max(y) > 0
        data = {'input': x, 'mask': y}  
        return data


#main
#get the file names
PathList =  path_list2()
no_patients = len(PathList[0])
inputs = PathList[0]
targets = PathList[1]
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


# fig  = plt.figure(figsize=(150,25))
# ax = []
# columns = 4
# rows = 2
# for i in range(0,no_patients):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].set_title(str(i+1))
#     PrintSlice(CTs[i], masks[i])
#     #projections(CTs[i], masks[i], order=[1,2,0])
# plt.show()

#projections(CTs[8], masks[8], order=[1,2,0])
#projections(CTs[9], masks[9], order=[1,2,0])
projections(CTs[1], masks[1], order=[1,2,0])
#PrintSlice(CTs[10], masks[10], show = True)

#%%
#save the preprocessed masks and cts for the dataset
save_preprocessed(CTs, masks, ids)


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