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

from neckNavigatorData import neckNavigatorDataset, get_data, head_augmentations
from neckNavigator import neckNavigator
#from NeckNavigatorHotMess import neckNavigator, neckNavigatorShrinkWrapped
from neckNavigatorTrainer import neckNavigator_trainer
from neckNavigatorUtils import k_fold_split_train_val_test
from neckNavigatorTrainerUtils import get_logger, get_number_of_learnable_parameters
from neckNavigatorTester import neckNavigatorTest1
from utils import setup_model, PrintSlice , projections, euclid_dis

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
    data_path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed_sphere.npz'
    checkpoint_dir = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1"
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

    # decide batch sizes
    train_BS = 1 #int(6 * args.GPUs)
    val_BS = 1 #int(6 * args.GPUs)
    train_workers = int(8)
    val_workers = int(4)

    # allocate ims to train, val and test
    dataset_size = len(inputs)
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num= 2)

    # dataloaders
    training_dataset = neckNavigatorDataset(inputs=inputs, targets=targets, image_inds = train_inds, transform = head_augmentations)#, transform = head_augmentations
    training_dataloader = DataLoader(dataset=training_dataset, batch_size= train_BS,  shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    
    validation_dataset = neckNavigatorDataset(inputs=inputs, targets=targets, image_inds = val_inds)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size= val_BS,  shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    test_dataset = neckNavigatorDataset(inputs = inputs, targets = targets, image_inds = test_inds)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # create model
    model = neckNavigator(filter_factor=2, targets = 1, in_channels = 1)
    #model = neckNavigator()
    #model = headHunter_multiHead_deeper(filter_factor=1)

    # put the model on GPU(s)
    device='cuda:1'
    # model.to(device)
    load_prev=False
    model=setup_model(model, checkpoint_dir, device, load_prev=load_prev)
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
 
    # Create the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.0005)

    # Create learning rate adjustment strategy
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Parallelize model
    #model = nn.DataParallel(model) #runs on multiple gpus if we want a larger batch size
    epoch = 0 
    iteration = 0

    if load_prev ==True:
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        epoch = state['epoch']
        iteration = state['num_iterations']
        print("starting epoch: ", epoch)
    
    # Create model trainer
    trainer = neckNavigator_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=training_dataloader, 
                                 val_loader=validation_dataloader, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=2, num_iterations = iteration, 
                                 num_epoch = epoch ,patience=20, iters_to_accumulate=4)
    
    # Start training
    trainer.fit()
#%%
    #testing
    tester = neckNavigatorTest1(model, test_dataloader, device)
    #test_results = tester
    C3s, segments, GTs = tester
    
    print("gt info: ", len(GTs))
    print(GTs.shape,)
    print("segs info: ", segments.shape)
    #print(segments[0].shape, len(segments))
    #print(C3s[0].shape)
    #c3 = C3s[0][0]
    #segment = segments[0][0]

    difference = euclid_dis(GTs, segments)
    print(difference)
    PrintSlice(C3s[0], segments[0], show=True)
    for j in range(0,4):
        projections(C3s[j],segments[j], order = [1,2,0])

    

    # fig  = plt.figure(figsize=(100,25))
    # ax = []
    # columns = 2
    # rows = 1
    # for i in range(0,rows*columns):
    #     ax.append(fig.add_subplot(rows, columns, i+1))
    #     ax[-1].set_title(str(i+1))
    #     PrintSlice(C3s[i], segments[i])
    #     #projections(c3s[0][i], segments[0][i], order=[1,2,0])
    # plt.savefig("slices.png")
    #plt.show()



    return
    
    
if __name__ == '__main__':
    main()