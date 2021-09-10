#UTILS FILE
#created: 09/09/2021

#imports
import numpy as np
import SimpleITK as sitk
from skimage.transform import rescale
from scipy.ndimage.measurements import center_of_mass
import torch
from functools import reduce
from operator import mul

from models import neckNavigator

#general utils functions
def GetTargetCoords(target):
  coords = center_of_mass(target) 
  return coords #z,x,y

def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""
    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)

###*** PRE-PROCESSING 1 ***###
#functions
def window_level(inp: np.array):
    window = 350
    level = 50
    vmax = level/2 + window
    vmin = level/2-window
    thresh = 1000
    inp[inp > thresh] = 0
    inp[inp > vmax] = vmax
    inp[inp < vmin] = vmin
    return inp

def flip(im):
    im = np.flip(im, axis=0)        # flip CC
    im = np.flip(im, axis=2)        # flip LR
    return im

def load(path):    
    # Load input and target
    x = sitk.ReadImage(path, imageIO="NiftiImageIO")
    #saving the spacing
    voxel_dim = [(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
    
    input_data = {'input': x, 'voxel_dims': np.array(voxel_dim).astype(np.float32)}  
    return input_data

class preprocessing_1():
    def __init__(self,
                 inputs: list,
                 transform = window_level, cropping = None, normalise = None):
        self.inputs = inputs
        self.transform = transform
        self.cropping = cropping
        self.normalise = normalise
        self.transform_list = []

    def __len__(self):
        return len(self.inputs)

    def transforms(self):
        #maybe save it here
        return np.array(self.transform_list)

    def __getitem__(self, index: int):
        # Select the sample
        input = self.inputs[index]
        # Load input and target
        x = input
        # check if flip required
        need_flip = False
        if x.GetDirection()[-1] == -1:
            need_flip = True
        x= sitk.GetArrayFromImage(x).astype(float)
        x-=1024 #worldmatch trickery

        # Preprocessing
        if need_flip == True:
            x= flip(x)

        if self.transform is not None:
            x = self.transform(x)
    
        if self.cropping is not None:
            x, crop_info = self.cropping(x)

        if self.normalise is not None:
            x = self.normalise(x)
        post_crop = GetTargetCoords(x)
        #downsampling to size -> [128,128,128]
        x = rescale(x, scale=((16/14),0.5,0.5), order=0, multichannel=False,  anti_aliasing=False)
        scale_info = np.array(post_crop)/np.array(GetTargetCoords(x))
        #save transforms
        transform_list_item = [need_flip, crop_info, scale_info]
        self.transform_list.append(np.array(transform_list_item))
    
        processed_data = {'input': x}  
        return processed_data

###*** NECK NAVIGATOR MODEL ***###
#MODEL SETUP
def setup_model(model, checkpoint_dir, device, load_prev = False, load_best = False, eval_mode = False):
  model.to(device)
  if load_prev == True:
    model.load_previous(checkpoint_dir, logger=None)
  if load_best == True:
    model.load_best(checkpoint_dir, logger = None)
  for param in model.parameters():
      param.requires_grad = True
  if eval_mode:
    model.eval()
  return model

def NeckNavigatorRun(model_dir, image, device): #put one image through
    model = setup_model(neckNavigator(), model_dir, device, load_best = True, eval_mode=True)
    CT = image
    input_image = torch.from_numpy(image)
    # creating channel dimension            
    input_image.unsqueeze(0)
    input_image = input_image.type(torch.FloatTensor).to(device)
    output = model(input_image)
    output = flat_softmax(output)
    test_output = output.squeeze().cpu().detach().numpy()
    pred = test_output.astype(np.float)
    return np.array(CT), np.array(pred)

###*** POST-PROCESSING 1 ***###
def mrofsnart(msks, transforms, shape = 128, test_inds = None):#transforms backwards
    x_arr,y_arr,z_arr = [],[],[]
    
    if test_inds is not None:
        transforms = [transforms[ind] for ind in test_inds]
        
    for i in range(len(msks)):
        #undo scale
        coords = GetTargetCoords(msks[i])
        #print(coords)
        mask = rescale(msks[i], scale=((14/16),2,2), order=0, multichannel=False,  anti_aliasing=False)
        coords = GetTargetCoords(mask)
        #undo crop
        #eg z crop [46,1] z=12  [[true, crop array, scale_info] <- crop[zmin, zmax, xmin,...],]
        z = coords[0] + transforms[i][1][0]
        x = coords[1]
        y = coords[2]
        #print("Undo Scale: ", coords)
        x += transforms[i][1][3]
        y += transforms[i][1][6]
        x_arr.append(x)
        #print("Undo Crop:", z, x, y)
        #undo flip if necessary
        if (transforms[i][0]==True):
            y_shape = transforms[i][1][8] -1
            y = y_shape - y
        y_arr.append(y)

        if (transforms[i][0]==True):
          z_shape = transforms[i][1][2] -1 
          z = z_shape - z
        #print("Final Coords: ", z,x,y)
        z_arr.append(z)
    return np.array(x_arr),np.array(y_arr),np.array(z_arr)

###*** SAVE NECK NAVIGATOR OUTPUT ***###
#slice number and sagital image and patient id

###*** PRE-PROCESSING 2 ***###

###*** MUSCLE MAPPER MODEL ***###

###*** POST-PROCESSING 2 ***###

###*** SAVE MUSCLE MAPPER OUTPUT ***###
#segment and patient ID and SMA/SMI and SMD