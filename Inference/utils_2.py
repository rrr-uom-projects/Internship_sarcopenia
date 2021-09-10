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
import cv2
import sklearn
from decimal import Decimal
import matplotlib.pyplot as plt

from models import neckNavigator

#general utils functions
def GetTargetCoords(target):
  coords = center_of_mass(target) 
  return coords #z,x,y

def GetSliceNumber(segment):
  slice_number = []
  weights = []
  max_range = len(segment)
  for x in range(0,max_range):
    seg_slice = segment[x,...]
    val = np.sum(seg_slice)
    if val != 0:
      slice_number.append(x)
      weights.append(val)
  slice_no = do_it_urself_round(np.average(slice_number, weights = weights), decimals=3)
  return slice_no

def do_it_urself_round(number, decimals = 0, is_array = False):
  float_num = number
  dec_num =  Decimal(str(number))
  Ddecimal = Decimal(str(decimals))
  round_num = round(dec_num, decimals)
  diff = np.abs(dec_num - round_num)
  round_diff = Decimal(str(0.5/(10**decimals)))
  if (diff == round_diff and dec_num >= round_num):
    round_num += (1/(10**(Ddecimal))) 
  if decimals == 0:
    return int(round_num)
  else:
    return float(round_num)

def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""
    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)

###*** LOAD ***###
def load(path):    
    # Load input and target
    x = sitk.ReadImage(path, imageIO="NiftiImageIO")
    #saving the spacing
    voxel_dim = [(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
    
    input_data = {'input': x, 'voxel_dims': np.array(voxel_dim).astype(np.float32)}  
    return input_data

###*** PRE-PROCESSING 1 ***###
def flip(im):
    im = np.flip(im, axis=0)        # flip CC
    im = np.flip(im, axis=2)        # flip LR
    return im

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

def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (plus clipping)"""
    inp = (np.nan_to_num(inp)).astype(np.float64)
    shape = inp.shape
    inp_new = np.round(sklearn.preprocessing.minmax_scale(inp.ravel(), feature_range=(0,1)), decimals = 10).reshape(shape)
    
    inp_out = inp_new
    return inp_out

def cropping(inp: np.ndarray):
    #working one but z axis crop needs improving
    x = inp
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
        
        inp = padded_arr
        z_coords = {"z_min": 0, "z_max": inp.shape[0]}
    
    x = inp[z_coords["z_min"]:z_coords["z_max"],x_min:x_max,y_min:y_max]
    cropped_info = [z_coords["z_min"], z_coords["z_max"], org_inp_size[0], x_min, x_max, org_inp_size[1], y_min, y_max, org_inp_size[2]]
    
    return x, np.array(cropped_info)

def preprocessing_1(image):
    x = image
    # check if flip required
    need_flip = False
    if x.GetDirection()[-1] == -1:
        need_flip = True
    x= sitk.GetArrayFromImage(x).astype(float)
    x-=1024 #worldmatch trickery

    # Preprocessing
    if need_flip == True:
        x= flip(x)

    x = window_level(x)
    x, crop_info = cropping(x)
    x = normalize_01(x)
    #downsampling to size -> [128,128,128]
    x = rescale(x, scale=((16/14),0.5,0.5), order=0, multichannel=False,  anti_aliasing=False)
    #save transforms
    transform_list = np.array([need_flip, crop_info])
    processed_data = {'input': x, 'transform': transform_list}  
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
    input_image = torch.from_numpy(image).type(torch.FloatTensor)
    # creating channel dimension            
    input_image = input_image.unsqueeze(0).unsqueeze(0)
    input_image = input_image.type(torch.FloatTensor).to(device)
    #print(input_image.shape)
    output = model(input_image)
    output = flat_softmax(output)
    test_output = output.squeeze().cpu().detach().numpy()
    pred = test_output.astype(np.float)
    return np.array(pred)

###*** POST-PROCESSING 1 ***###
def mrofsnart(msk, transforms, shape = 128):#transforms backwards
    #undo scale
    coords = GetTargetCoords(msk)
    #print(coords)
    mask = rescale(msk, scale=((14/16),2,2), order=0, multichannel=False,  anti_aliasing=False)
    coords = GetTargetCoords(mask)
    #undo crop
    #eg z crop [46,1] z=12  [[true, crop array] <- crop[zmin, zmax, org_size, xmin,...],]
    z = coords[0] + transforms[1][0]
    x = coords[1]
    y = coords[2]
    #print("Undo Scale: ", coords)
    x += transforms[1][3]
    y += transforms[1][6]
    #print("Undo Crop:", z, x, y)
    #undo flip if necessary
    if (transforms[0]==True):
        y_shape = transforms[1][8] -1
        y = y_shape - y

    if (transforms[0]==True):
        z_shape = transforms[1][2] -1 
        z = z_shape - z
    #print("Final Coords: ", z,x,y)
    return do_it_urself_round(x,1),do_it_urself_round(y,1),do_it_urself_round(z,1)

###*** SAVE NECK NAVIGATOR OUTPUT ***###
#slice number and sagital image and patient id
def arrange(input, ax, proj_order):
    #to return the projection in whatever order.
    av = np.average(input, axis = ax)
    mx = np.max(input, axis = ax)
    std = np.std(input, axis=ax)
    ord_list = [mx,av,std]
    ord_list[:] = [ord_list[i] for i in proj_order]
    out = np.stack((ord_list), axis=2)
    return out

def base_projections(inp, msk):
  axi,cor,sag = 0,1,2
  proj_order = [2,1,0]
  axial = arrange(inp, axi, proj_order)
  coronal = arrange(inp, cor, proj_order)
  sagital = arrange(inp, sag, proj_order)
  ax_mask = np.max(msk, axis = axi)
  cor_mask = np.max(msk, axis = cor)
  sag_mask = np.max(msk, axis = sag)
  #flip right way up
  coronal = coronal[::-1]
  sagital = sagital[::-1]
  cor_mask = cor_mask[::-1]
  sag_mask = sag_mask[::-1]
  image = (axial,coronal,sagital)
  mask = (ax_mask,cor_mask, sag_mask)
  return image, mask

def display_net_test(inps, msks, id, shape = 128, z = None):
    coords_pred = GetTargetCoords(msks)  
    #make the figure
    fig = plt.figure()
    image, pred = base_projections(inps, msks)
    slice_pred = shape - do_it_urself_round(GetSliceNumber(msks)) #as projections have different origin
    plt.title('C3 pred: ' + id)
    plt.imshow(image[2])
    #plt.imshow(pred[2], cmap = 'cool', alpha=0.5) #maybe take this out
    plt.axhline(slice_pred, linewidth=2, c='y')
    if z is None: slice_no = slice_pred
    else: slice_no = do_it_urself_round(z)
    plt.text(0, slice_pred-5, "C3: "+ str(slice_no), color='w')
    plt.scatter(coords_pred[1], (128 - coords_pred[0]), c = 'r', s=20) #y,z
    plt.axis('off')
    plt.show()
    return fig
###*** PRE-PROCESSING 2 ***###

###*** MUSCLE MAPPER MODEL ***###

###*** POST-PROCESSING 2 ***###

###*** SAVE MUSCLE MAPPER OUTPUT ***###
#segment and patient ID and SMA/SMI and SMD