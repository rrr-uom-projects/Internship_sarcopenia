#27/07/2021
# Hermione Warr and Olivia Murray
# consensually stolen and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter
# neckNavigator utils

#imports
import numpy as np
from itertools import cycle
import torch
import shutil
import os
from torch.utils import data

# def k_fold_split_train_val_test(dataset_size, fold_num):
#     k = int(fold_num-1)
#     #train_ims, val_ims, test_ims = 192, 24, 22
#     train_ims, val_ims, test_ims = round(0.8*dataset_size), round(0.1*dataset_size), round(0.1*dataset_size)
#     print(train_ims,val_ims,test_ims, dataset_size)
#     assert(train_ims+val_ims+test_ims == dataset_size)
#     train_inds, val_inds, test_inds = [], [], []
#     # initial shuffle
#     np.random.seed(2305)
#     shuffled_ind_list = np.random.permutation(dataset_size)
#     # allocate dataset indices based upon the fold number --> not the prettiest or most efficient implementation, but functional
#     cyclic_ind_list = cycle(shuffled_ind_list)
#     for i in range(k*test_ims):
#         next(cyclic_ind_list)   # shift start pos
#     for i in range(test_ims):
#         test_inds.append(next(cyclic_ind_list))
#     for i in range(train_ims):
#         train_inds.append(next(cyclic_ind_list))
#     for i in range(val_ims):
#         val_inds.append(next(cyclic_ind_list))
#     return train_inds, val_inds, test_inds

# def k_fold_split_testset_inds(dataset_size, fold_num):
#     k = int(fold_num-1)
#     train_ims, val_ims, test_ims = 192, 24, 22
#     assert(train_ims+val_ims+test_ims == dataset_size)
#     train_inds, val_inds, test_inds = [], [], []
#     # initial shuffle
#     np.random.seed(2305)
#     shuffled_ind_list = np.random.permutation(dataset_size)
#     # allocate dataset indices based upon the fold number --> not the prettiest or most efficient implementation, but functional
#     cyclic_ind_list = cycle(shuffled_ind_list)
#     for i in range(k*test_ims):
#         next(cyclic_ind_list)   # shift start pos
#     for i in range(test_ims):
#         test_inds.append(next(cyclic_ind_list))
#     for i in range(train_ims):
#         train_inds.append(next(cyclic_ind_list))
#     for i in range(val_ims):
#         val_inds.append(next(cyclic_ind_list))
#     return test_inds

class RunningAverage:
    # Computes and stores the average
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)
    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)