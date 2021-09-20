#Created: 09/09/2021
#by Hermione and Olivia
#inference script
#Script: give path to nifti file (one image) runs pipeline. Writes out image, mask, excel info, into a folder. 
# Sanity check folder. Folder for masks and for stats. SMA and SMD and slice number. 
#need to load best model weights

#imports
from matplotlib.pyplot import savefig
import pandas as pd
from sklearn import preprocessing

from utils_2 import load, get_patient_id, postprocessing_2, preprocessing_1, NeckNavigatorRun, mrofsnart
from utils_2 import preprocessing_2, MuscleMapperRun, postprocessing_2, save_figs, append_df_to_excel

###*** GLOBAL VARIABLES ***###
#paths
path = '/home/hermione/t/Donal/JP_HNC'
#path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/images'
#'D:/data/Alex/HeadAndNeckData/Packs_UKCatsFeedingTube'
NN_model_weights_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs_fold1'
MM_model_weights_path = "/home/hermione/Documents/Internship_sarcopenia/Inference/MM3_model_state_dict_fold6.pt"
sanity_check_folder = '/home/hermione/Documents/Internship_sarcopenia/Inference/sanity_check/'
xl_writer = sanity_check_folder + 'skeletal_muscle_info.xlsx'
#constants
window = 350
level = 50

paths, patient_ids = get_patient_id(path)

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
    for patient in range(len(paths)):
        id = patient_ids[patient]
        print("\nPatient ", patient, "id ", id)

        ###*** LOAD ***###
        try:
            input_data = load(paths[patient])
        except:
            continue
        image = input_data['input']
        voxel_dims = input_data['voxel_dims']

        ###*** PRE-PROCESSING 1 ***###
        preprocessing_info = preprocessing_1(image)
        processed_ct = preprocessing_info['input']
        transforms = preprocessing_info['transform']
        
        ###*** RUN NECK NAVIGATOR MODEL ***###
        NN_pred = NeckNavigatorRun(NN_model_weights_path, processed_ct, device)#load_best = true
        
        ###*** POST-PROCESSING 1 ***###
        #extract predicted slice number (and other coords)
        x,y,z = mrofsnart(NN_pred, transforms)

        ###*** SAVE NECK NAVIGATOR OUTPUT ***###
        #save slice number and sagital image and patient id <- path end
        print("Slice No:", z)
        
        ###*** PRE-PROCESSING 2 ***###
        #hmm the scale for the other model might be and issue maybe save the image before the scale is applied and use that.
        #or might have to use the loaded in image and preprocessing again to select the right slice.
        preprocess2_info = preprocessing_2(z, image)
        processed_slice = preprocess2_info["slice"]
        bone_mask = preprocess2_info["bone"]


        ###*** MUSCLE MAPPER MODEL ***###
        MM_segment = MuscleMapperRun(processed_slice, MM_model_weights_path, device)
        import numpy as np
        print(MM_segment.shape, np.unique(MM_segment))

        ###*** POST-PROCESSING 2 ***###
        #use image here not processed.remove bone from segmentation
        SMA, SMD = postprocessing_2(image, z, MM_segment, bone_mask, voxel_dims)
        print("SMA: ",SMA, "SMD", SMD)

        ###*** SAVE MUSCLE MAPPER OUTPUT ***###
        #segment and patient ID and SMA/SMI and SMD to excel
        save_figs(processed_ct, processed_slice, MM_segment, NN_pred, sanity_check_folder, id = id)

        df = pd.DataFrame({"IDs": [id], "SMA": [SMA] ,"SMD":[SMD]})
        append_df_to_excel(xl_writer, df, sheet_name='SM_data', index = False)
    
    return

if __name__ == '__main__':
    main()
