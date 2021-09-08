# 27/07/2021
# Hermione Warr and Olivia Murray
# consensually stolen and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter
# train_neckNavigator.py

#imports
from numpy.lib.function_base import _diff_dispatcher
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import os
#from scipy.ndimage import distance_transform_edt as dist_xfm
import argparse as ap
import pandas as pd

from neckNavigatorData import neckNavigatorDataset, head_augmentations
from neckNavigator import neckNavigator
#from NeckNavigatorHotMess import neckNavigator
from neckNavigatorTrainer import neckNavigator_trainer
from neckNavigatorTrainerUtils import k_fold_cross_val, get_logger, get_number_of_learnable_parameters, dataset_TVTsplit
from neckNavigatorTester import neckNavigatorTest2

from utils import setup_model, get_data, mrofsnart

def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for 3D location-finding network \"headhunter\"")
    parser.add_argument("--targets", default=1, type=int, help="The number of targets")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--GPUs", choices=[1,2], type=int, default=1, help="Number of GPUs to use")
    global args
    args = parser.parse_args()


def main():
    #main
    
    # get args
    setup_argparse()
    global args

    #livs paths
    data_path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed_Tsphere.npz'
    checkpoint = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/model_ouputs"
    save_path =  '/home/olivia/Documents/Internship_sarcopenia/locating_c3/fold_info/c3_loc.xlsx'
    #herms paths


    #data_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_Tgauss.npz'
    #checkpoint = "/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs"
    #save_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/fold_info/c3_loc.xlsx'#' +str(i) + '

    
    # Create main logger
    logger = get_logger('NeckNavigator_Training')

    # get data
    data = get_data(data_path)
    inputs = data[0]
    targets = data[1]
    ids = data[2]
    transforms = data[3]
    org_slices = data[4]
    voxel_dims = data[5]


    # decide batch sizes
    train_BS = 1 #int(6 * args.GPUs)
    val_BS = 1 #int(6 * args.GPUs)

    train_workers = int(8)
    val_workers = int(4)


    # allocate ims to train, val and test
    dataset_size = len(inputs)
    train_array, test_array = k_fold_cross_val(dataset_size, num_splits = 5)
    
    #######*** K FOLD CROSS VALIDATION LOOP ***#######
    for i in range(0,5):

        checkpoint_dir = checkpoint + "_fold" + str(i+1)
        try:
            os.makedirs(checkpoint_dir)
        except OSError: #if already exists
            pass

        ###*** CREATE DATALOADERS ***###
        val_split, train_split = np.split(train_array[i], [round(0.2*len(train_array[i]))], axis= 0)#[16]
        train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = dataset_TVTsplit(inputs, targets, train_split, val_split, test_array[i])
        
        training_dataset = neckNavigatorDataset(inputs=train_inputs, targets=train_targets, transform = head_augmentations)
        training_dataloader = DataLoader(dataset=training_dataset, batch_size= train_BS,  shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))#worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
    
        validation_dataset = neckNavigatorDataset(inputs=val_inputs, targets=val_targets)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size= val_BS,  shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

        test_dataset = neckNavigatorDataset(inputs = test_inputs, targets = test_targets)
        test_dataloader = DataLoader(dataset= test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

        ###*** INITIALISE MODEL ***###
        model = neckNavigator()

        device = 'cuda:1'



        load_prev=False
        model = setup_model(model, checkpoint_dir, device)

        # Log the number of learnable parameters
        logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    
        # Create the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)

        # Create learning rate adjustment strategy
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Parallelize model
        #model = nn.DataParallel(model) #runs on multiple gpus if we want a larger batch size
        
        epoch = 0 
        iteration = 0

        if load_prev == True:
            state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
            epoch = state['epoch']
            iteration = state['num_iterations']
            print("starting epoch: ", epoch)
        
        # Create model trainer
        trainer = neckNavigator_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=training_dataloader, 
                                    val_loader=validation_dataloader, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=300, num_iterations = iteration, 
                                    num_epoch = epoch, patience=50, iters_to_accumulate=4)

        
        ###*** TRAINING ***###
        trainer.fit()

        #####*** TESTING ***#####
        tester = neckNavigatorTest2(checkpoint_dir, test_dataloader, device)
        C3s, segments, GTs = tester

        test_inds = test_array[i]
        test_org_slices = org_slices[test_inds]
        test_vox_dims = voxel_dims[test_inds]
        test_processing_info = transforms[test_inds]
        test_ids = ids[test_inds]

        ####*** SAVE MODEL PREDICTIONS ***####
        pred_save_path = checkpoint_dir + "/trained_net_tests.npz"
        np.savez(pred_save_path, images = C3s, preds = segments, gts = GTs, index = test_inds, ids = test_ids, 
            test_processing = test_processing_info, test_preprocessed_gt_slices = test_org_slices, test_vox_dims = test_vox_dims)


    return
        
    
if __name__ == '__main__':
    main()
