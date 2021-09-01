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
#from scipy.stats import norm
#from scipy.ndimage import distance_transform_edt as dist_xfm
#import random
#import sys
import argparse as ap
#import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from neckNavigatorData import neckNavigatorDataset, head_augmentations#, get_data
from neckNavigator import neckNavigator
#from NeckNavigatorHotMess import neckNavigator, neckNavigatorShrinkWrapped
from neckNavigatorTrainer import neckNavigator_trainer
from neckNavigatorUtils import k_fold_split_train_val_test
from neckNavigatorTrainerUtils import get_logger, get_number_of_learnable_parameters, dataset_TVTsplit
from neckNavigatorTester import neckNavigatorTest2

from utils import setup_model, PrintSlice , projections, euclid_dis, get_data
from utils import k_fold_cross_val, slice_preds, GetSliceNumber, GetTargetCoords
from utils import mrofsnart


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

    # decide file paths
    #livs paths
    data_path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed_Tgauss.npz'
    


    #herms paths
    #data_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_gauss.npz'
    #checkpoint_dir = "/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs"



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

    train_workers = 0 #int(8)
    val_workers = 0 #int(4)


    # allocate ims to train, val and test
    dataset_size = len(inputs)
    #train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num= 2)
    #train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = dataset_TVTsplit(inputs, targets, train_inds, val_inds, test_inds)
    train_array, test_array = k_fold_cross_val(dataset_size, num_splits = 5)
    
    for i in range(0,5):
    # dataloaders

        checkpoint_dir = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/fold" + str(i)

        val_spli, train_spli = np.split(train_array[i], [16], axis= 0)
        train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = dataset_TVTsplit(inputs, targets, train_spli, val_spli, test_array[i])
        training_dataset = neckNavigatorDataset(inputs=train_inputs, targets=train_targets, transform = head_augmentations)#, transform = head_augmentations

        training_dataloader = DataLoader(dataset=training_dataset, batch_size= train_BS,  shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=None)#worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
    
        validation_dataset = neckNavigatorDataset(inputs=val_inputs, targets=val_targets)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size= val_BS,  shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=None)


        test_dataset = neckNavigatorDataset(inputs = test_inputs, targets = test_targets)
        test_dataloader = DataLoader(dataset= test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

        #torch.save(test_dataloader, dataloader_dir, pickle_protocol= pickle.HIGHEST_PROTOCOL)
        # create model
        model = neckNavigator()

        # put the model on GPU(s)
        device = 'cuda:1'
        # model.to(device)
        load_prev=False
        model=setup_model(model, checkpoint_dir, device, load_prev= False, load_best = load_prev)
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
                                    val_loader=validation_dataloader, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=1, num_iterations = iteration, 

                                    num_epoch = epoch ,patience=50, iters_to_accumulate=4)

        
        # Start training
        trainer.fit()


        #testing
        #model = setup_model(model, checkpoint_dir, device, load_prev=True, eval_mode=True)
        tester = neckNavigatorTest2(checkpoint_dir, test_dataloader, device)
        C3s, segments, GTs = tester
        difference = euclid_dis(GTs, segments)

        slice_no_preds = slice_preds(segments)
        slice_no_gts = slice_preds(GTs)
        slice_no_gts_test = []

        for j in range(len(GTs)): #sanity check
            slice_no_gt = GetSliceNumber(GTs[j])
            slice_no_gts_test.append(slice_no_gt)





        x,y,z = mrofsnart(slice_no_preds, transforms, test_inds =test_array)
        df = pd.DataFrame({"IDs": ids[test_array[i]], "Slice_Numbers": slice_no_preds, "PostProcess Slice numbers": z, "GT org slices": test_org_slices, "GT T Slices": slice_no_gts, "Distances": difference})
        save_path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/fold' +str(i) + '/c3_loc.xlsx'
        df.to_excel(excel_writer = save_path, index=False,
                sheet_name="data")
        
        

        
    return
        
    
if __name__ == '__main__':
    main()
