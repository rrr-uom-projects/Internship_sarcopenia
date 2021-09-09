#UTILS FILE
#created: 09/09/2021

#imports
import numpy as np
import SimpleITK as sitk
from skimage.transform import rescale

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

#classes
class load():
    def __init__(self,
                 inputs: list,
                 transform = window_level, normalise = None ):
        self.inputs = inputs
    
    def __len__(self):
        return len(self.inputs)

    def voxel_dims(self):
        return np.array(self.voxel_dims_list)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        # Load input and target
        x = sitk.ReadImage(input_ID, imageIO="NiftiImageIO")
        #saving the spacing
        voxel_dim = [(x.GetSpacing())[0],(x.GetSpacing())[1],(x.GetSpacing())[2]]
        self.voxel_dims_list.append(np.array(voxel_dim).astype(np.float32))
        
        input_data = {'input': x}  
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

        #downsampling to size -> [128,128,128]
        x = rescale(x, scale=((16/14),0.5,0.5), order=0, multichannel=False,  anti_aliasing=False)
        #scale_info = np.array(post_crop)/np.array(GetTargetCoords(y))
        #print("Scale info: ", scale_info)
        #save transforms and gt slice
        transform_list_item = [need_flip, crop_info, scale_info]
        self.transform_list.append(np.array(transform_list_item))
        assert np.min(y) >= 0
        assert np.max(y) > 0
        data = {'input': x, 'mask': y}  
        return data

###*** NECK NAVIGATOR MODEL ***###

###*** POST-PROCESSING 1 ***###

###*** SAVE NECK NAVIGATOR OUTPUT ***###
#slice number and sagital image and patient id

###*** PRE-PROCESSING 2 ***###

###*** MUSCLE MAPPER MODEL ***###

###*** POST-PROCESSING 2 ***###

###*** SAVE MUSCLE MAPPER OUTPUT ***###
#segment and patient ID and SMA/SMI and SMD