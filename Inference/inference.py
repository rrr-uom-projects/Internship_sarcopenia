#Created: 09/09/2021
#by Hermione and Olivia
#inference script
#Script: give path to nifti file (one image) runs pipeline. Writes out image, mask, excel info, into a folder. 
# Sanity check folder. Folder for masks and for stats. SMA and SMD and slice number. 
#need to load best model weights

#imports
from models import neckNavigator

###*** GLOBAL VARIABLES ***###
#paths
path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/images/100182405.nii.gz'
model_weights_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs_fold1/best_checkpoint.pytorch'
sanity_check_folder = '/home/hermione/Documents/Internship_sarcopenia/Inference/sanity_check/'
#constants
window = 350
level = 50

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
    ###*** PRE-PROCESSING 1 ***###
    #load in and preprocess <- hmm seperate?. save voxel dims for calculating SMA later.

    ###*** RUN NECK NAVIGATOR MODEL ***###
    #load in model weigts and run model over one image. need dataloader

    ###*** POST-PROCESSING 1 ***###
    #extract predicted slice number (and other coords)
    
    ###*** SAVE NECK NAVIGATOR OUTPUT ***###
    #save slice number and sagital image and patient id

    ###*** PRE-PROCESSING 2 ***###
    #hmm the scale for the other model might be and issue maybe save the image before the scale is applied and use that.
    #or might have to use the loaded in image and preprocessing again to select the right slice.
    
    ###*** MUSCLE MAPPER MODEL ***###

    ###*** POST-PROCESSING 2 ***###

    ###*** SAVE MUSCLE MAPPER OUTPUT ***###
    #segment and patient ID and SMA/SMI and SMD to excel
    
    return

if __name__ == '__main__':
    main()
