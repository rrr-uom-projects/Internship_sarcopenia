# consensually sloten and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter
# train_headHunter.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt as dist_xfm
import random
import sys
import os
import argparse as ap
from kornia.geometry import transform
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential 
from kornia.utils import image_to_tensor, tensor_to_image

from dataset_3d import Segmentation3DDataset, get_data, head_augmentations
from EdHead import headHunter
from EdHeadTrain import headHunter_trainer
from EdHeadUtils import k_fold_split_train_val_test
from EdHeadTrainerUtils import get_logger, get_number_of_learnable_parameters, getFiles, windowLevelNormalize

# imagedir = "/data/Jigsaw3D/headHunter_data/cropped_n_scaled/"
# CoM_targets = np.load("/data/Jigsaw3D/headHunter_data/CoM_targets.npy")
# #imagedir = "/data/Jigsaw3D/spineSeeker_data/cropped_n_scaled/"
# #CoM_targets = np.load("/data/Jigsaw3D/spineSeeker_data/spine_points_list.npy")

def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for 3D location-finding network \"headhunter\"")
    parser.add_argument("--targets", default=1, type=int, help="The number of targets")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--GPUs", choices=[1,2], type=int, default=1, help="Number of GPUs to use")
    global args
    args = parser.parse_args()

def main():
    # get args
    setup_argparse()
    global args

    # decide checkpoint directory

    checkpoint_dir = "/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs/"

    # Create main logger
    logger = get_logger('HeadHunter_Training')

    # Create the model
    model = headHunter(filter_factor=2, targets=args.targets, in_channels=1)
    #model = spineSeeker()
    for param in model.parameters():
        param.requires_grad = True

    # put the model on GPU(s)
    device='cuda:0'
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    
    train_BS = 1 #int(6 * args.GPUs)
    val_BS = 1 #int(6 * args.GPUs)
    train_workers = int(8)
    val_workers = int(4)
#main
    data = get_data()
    inputs = data[0]
    targets = data[1]
    ids = data[2]




    # allocate ims to train, val and test
    dataset_size = len(inputs)
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num= 2)

    # dataloaders
    training_dataset = Segmentation3DDataset(inputs=inputs, targets=targets, image_inds = train_inds, transform = head_augmentations)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size= train_BS,  shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    
    validation_dataset = Segmentation3DDataset(inputs=inputs, targets=targets, image_inds = val_inds)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size= val_BS,  shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))


    # # allocate ims to train, val and test
    # dataset_size = len(sorted(getFiles(imagedir)))
    # train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num=args.fold_num)

    # # Create them dataloaders
    # train_data = headHunter_Dataset(imagedir=imagedir, image_inds=train_inds, shift_augment=True, flip_augment=True)
    # train_loader = DataLoader(dataset=train_data, batch_size=train_BS, shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    # val_data = headHunter_Dataset(imagedir=imagedir, image_inds=val_inds, shift_augment=True, flip_augment=True)
    # val_loader = DataLoader(dataset=val_data, batch_size=val_BS, shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # Create the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.005)

    # Create learning rate adjustment strategy
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=175, verbose=True)

    # Parallelize model
    model = nn.DataParallel(model)
    
    # Create model trainer
    trainer = headHunter_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=training_dataloader, 
                                 val_loader=validation_dataloader, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=1000, patience=500, iters_to_accumulate=1)
    
    # Start training
    trainer.fit()

    # Used to test here -> relocated to separate script to test end-to-end :)
    # complete
    return
    
# class headHunter_Dataset(data.Dataset):
#     def __init__(self, imagedir, image_inds, heatmap=True, shift_augment=True, flip_augment=True):
#         self.imagedir = imagedir
#         self.availableImages = [sorted(getFiles(imagedir))[ind] for ind in image_inds]
#         self.targets = np.array([CoM_targets[ind] for ind in image_inds])
#         self.heatmap = heatmap
#         self.shifts = shift_augment
#         self.flips = flip_augment
#         self.gaussDist = norm(scale=10)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#            idx = idx.tolist()
#         imageToUse = self.availableImages[idx]
#         ct_im = np.load(os.path.join(self.imagedir, imageToUse))
#         target = self.targets[idx].copy()
#         # Augmentations
#         if self.shifts or self.flips:
#             if self.shifts:
#                 # find shift values
#                 cc_shift, ap_shift, lr_shift = random.randint(-2,2), random.randint(-4,4), random.randint(-4,4)
#                 # pad for shifting into
#                 ct_im = np.pad(ct_im, pad_width=((2,2),(4,4),(4,4)), mode='constant')
#                 # crop to complete shift
#                 ct_im = ct_im[2+cc_shift:50+cc_shift,4+ap_shift:124+ap_shift,4+lr_shift:124+lr_shift]
#                 # nudge the target to match the shift -> will work with single targets and multi-targets (reason for the '...,')
#                 target[...,0] -= cc_shift
#                 target[...,1] -= ap_shift
#                 target[...,2] -= lr_shift
#             if self.flips:
#                 if random.choice([True, False]):
#                     # implement LR flip
#                     ct_im = np.flip(ct_im, axis=2).copy()
#                     target[...,2] = 119 - target[...,2]
#                     #target = np.flip(target, axis=0).copy()    ## <-- special case for the parotids!
#         '''
#         # perform window-levelling here, create 3 channels
#         ct_im3 = np.zeros(shape=(3, ct_im.shape[0], ct_im.shape[1], ct_im.shape[2]))
#         ct_im3[0] = windowLevelNormalize(ct_im, level=1064, window=350)
#         ct_im3[1] = windowLevelNormalize(ct_im, level=1064, window=80)
#         ct_im3[2] = windowLevelNormalize(ct_im, level=1624, window=2800)
#         '''
#         ct_im3 = np.zeros(shape=(1,)+ct_im.shape)
#         ct_im3[0] = windowLevelNormalize(ct_im, level=1624, window=1600)
#         # now convert target to heatmap target
#         if self.heatmap:
#             '''
#             h_target = np.ones(shape=ct_im.shape)
#             h_target[int(round(target[0])), int(round(target[1])), int(round(target[2]))] = 0
#             h_target = dist_xfm(h_target, sampling=(2,1,1))
#             h_target = self.gaussDist.pdf(h_target)
#             h_target /= np.max(h_target)
#             return {'ct_im': ct_im3, 'target': target, 'h_target': h_target[np.newaxis]}
#             '''
#             # new off grid heatmap generation - generalised to multiple targets
#             if target.shape[0] == 1:
#                 t = np.indices(dimensions=ct_im.shape).astype(float)
#                 dist_map = np.sqrt(np.sum([np.power((2*(t[0] - target[0])), 2), np.power((t[1] - target[1]), 2), np.power((t[2] - target[2]), 2)], axis=0))
#                 h_target = self.gaussDist.pdf(dist_map)
#                 h_target /= np.max(h_target)
#                 return {'ct_im': ct_im3, 'target': target, 'h_target': h_target[np.newaxis]}
#             else:
#                 n_targets = target.shape[0]
#                 multi_h_target = np.zeros(shape=(n_targets,)+ct_im.shape)
#                 for loc_idx in range(n_targets):
#                     t = np.indices(dimensions=ct_im.shape).astype(float)
#                     dist_map = np.sqrt(np.sum([np.power((2*(t[0] - target[loc_idx, 0])), 2), np.power((t[1] - target[loc_idx, 1]), 2), np.power((t[2] - target[loc_idx, 2]), 2)], axis=0))
#                     h_target = self.gaussDist.pdf(dist_map)
#                     h_target /= np.max(h_target)
#                     multi_h_target[loc_idx] = h_target
#                 return {'ct_im': ct_im3, 'target': target, 'h_target': multi_h_target} # <- already has the required additional channels axis
#         return {'ct_im': ct_im, 'target': target}
        
#     def __len__(self):
#         return len(self.availableImages)
    
if __name__ == '__main__':
    main()