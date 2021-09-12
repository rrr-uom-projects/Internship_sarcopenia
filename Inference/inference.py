#Created: 09/09/2021
#by Hermione and Olivia
#inference script
#Script: give path to nifti file (one image) runs pipeline. Writes out image, mask, excel info, into a folder. 
# Sanity check folder. Folder for masks and for stats. SMA and SMD and slice number. 
#need to load best model weights

#imports
#from models import neckNavigator
from matplotlib.pyplot import savefig
from sklearn import preprocessing
from utils_2 import load, postprocessing_2, preprocessing_1, NeckNavigatorRun, mrofsnart, display_net_test
from utils_2 import preprocessing_2, MuscleMapperRun, postprocessing_2, display_slice, save_figs

###*** GLOBAL VARIABLES ***###
#paths
path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/images/100182405.nii.gz'
#path = 'D:/data/Alex/HeadAndNeckData/Packs_UKCatsFeedingTube' #Alex's data
NN_model_weights_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs_fold1'
MM_model_weights_path = "/home/hermione/Documents/Internship_sarcopenia/Inference/model_state_dict_300_FL_testing.pt"
sanity_check_folder = '/home/hermione/Documents/Internship_sarcopenia/Inference/sanity_check/'
#constants
window = 350
level = 50

#patient_id = 

device = 'cuda:1'

def main():  
    """
    Run the full segmentation workflow:
        - Load those images in directory
            - Crop, flip, window/level, normalise and scale the images as they are loaded
        - Run inference 1
        - Extract c3 slice
        - Run inference 2
        - Write segmentation masks to the output directory
            - unflip, resample on the way
    """  
    ###*** LOAD ***###
    input_data = load(path)
    image = input_data['input']
    voxel_dims = input_data['voxel_dims']
    ###*** PRE-PROCESSING 1 ***###
    #load in and preprocess <- hmm seperate?. save voxel dims for calculating SMA later.
    preprocessing_info = preprocessing_1(image)
    processed_ct = preprocessing_info['input']
    transforms = preprocessing_info['transform']
    ###*** RUN NECK NAVIGATOR MODEL ***###
    #load in model weigts and run model over one image. need dataloader
    NN_pred = NeckNavigatorRun(NN_model_weights_path, processed_ct, device)#load_best = true
    ###*** POST-PROCESSING 1 ***###
    #extract predicted slice number (and other coords)
    x,y,z = mrofsnart(NN_pred, transforms)
    ###*** SAVE NECK NAVIGATOR OUTPUT ***###
    #save slice number and sagital image and patient id <- path end
    # sagital_fig = display_net_test(processed_ct, pred, id = 'test', z= z)
    print(z)
    
    ###*** PRE-PROCESSING 2 ***###
    #hmm the scale for the other model might be and issue maybe save the image before the scale is applied and use that.
    #or might have to use the loaded in image and preprocessing again to select the right slice.
    processed_slice, bone_mask = preprocessing_2(z, image)
    print(processed_slice.shape)

    ###*** MUSCLE MAPPER MODEL ***###
    MM_segment = MuscleMapperRun(processed_slice, bone_mask, MM_model_weights_path, device)
    import numpy as np
    print(MM_segment.shape, np.unique(MM_segment))
    ###*** POST-PROCESSING 2 ***###
    #use image here not processed.remove bone from segmentation
    SMA, SMD = postprocessing_2(image, z, MM_segment, voxel_dims)
    print("SMA: ",SMA, "SMD", SMD)
    ###*** SAVE MUSCLE MAPPER OUTPUT ***###
    #segment and patient ID and SMA/SMI and SMD to excel
    # slice_fig = display_slice(processed_slice, segment)
    save_figs(processed_ct, processed_slice, MM_segment, NN_pred, sanity_check_folder, id ='test')
    return

if __name__ == '__main__':
    main()
