# 27/07/2021
# Hermione Warr and Olivia Murray
# consensually stolen and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter
# train_neckNavigator.py

#imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
#from scipy.stats import norm
#from scipy.ndimage import distance_transform_edt as dist_xfm
#import random
#import sys
#import os
import argparse as ap
#import tensorflow as tf
import matplotlib.pyplot as plt

from neckNavigatorData import neckNavigatorDataset, get_data, head_augmentations
from neckNavigator import neckNavigator, headHunter_multiHead_deeper
#from NeckNavigatorHotMess import neckNavigator, neckNavigatorShrinkWrapped
from neckNavigatorTrainer import neckNavigator_trainer
from neckNavigatorUtils import k_fold_split_train_val_test
from neckNavigatorTrainerUtils import get_logger, get_number_of_learnable_parameters, getFiles, windowLevelNormalize
from neckNavigatorTester import neckNavigatorTest
from utils import PrintSlice , GetSliceNumber, projections

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
    #data_path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed(3).npz'
    #checkpoint_dir = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1"
    #herms paths
    data_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed.npz'
    checkpoint_dir = "/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs"

    # Create main logger
    logger = get_logger('NeckNavigator_Training')

    # get data
    data = get_data(data_path)
    inputs = data[0]
    targets = data[1]
    ids = data[2]

    # decide batch sizes
    train_BS = 1 #int(6 * args.GPUs)
    val_BS = 1 #int(6 * args.GPUs)
    train_workers = int(8)
    val_workers = int(4)

    # allocate ims to train, val and test
    dataset_size = len(inputs)
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num= 2)

    # dataloaders
    training_dataset = neckNavigatorDataset(inputs=inputs, targets=targets, image_inds = train_inds, transform = head_augmentations)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size= train_BS,  shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    
    validation_dataset = neckNavigatorDataset(inputs=inputs, targets=targets, image_inds = val_inds)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size= val_BS,  shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    test_dataset = neckNavigatorDataset(inputs = inputs, targets = targets, image_inds = test_inds)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # create model
    model = neckNavigator(filter_factor=2, targets= 1, in_channels=1)
    #model = neckNavigator()
    #model = headHunter_multiHead_deeper(filter_factor=1)
    for param in model.parameters():
        param.requires_grad = True

    # put the model on GPU(s)
    device='cuda:0'
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
 
    # Create the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.005)

    # Create learning rate adjustment strategy
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=175, verbose=True)

    # Parallelize model
    model = nn.DataParallel(model)
    
    # Create model trainer
    trainer = neckNavigator_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=training_dataloader, 
                                 val_loader=validation_dataloader, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=1, patience=2, iters_to_accumulate=1)
    
    # Start training
    trainer.fit()
    tester = neckNavigatorTest(model, test_dataloader)
    c3s, segments = tester[0], tester[1]

    print(segments[0].shape, len(segments))
    print(c3s[0].shape)
    c3 = c3s[0][0]
    segment = segments[0][0]
    #PrintSlice(c3, segment)

    projections(c3,segment, order = [1,2,0])
    fig  = plt.figure(figsize=(150,25))
    ax = []
    columns = 4
    rows = 2
    for i in range(0,len(c3s[0])):
        ax.append(fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title(str(i+1))
        PrintSlice(c3s[0][i], segments[0][i])
        #projections(c3s[0][i], segments[0][i], order=[1,2,0])
    plt.show()

    return
    
    
if __name__ == '__main__':
    main()