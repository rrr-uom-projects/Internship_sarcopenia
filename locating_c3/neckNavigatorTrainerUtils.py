# 27/07/2021
# Hermione Warr and Olivia Murray
# consensually stolen and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter

#imports
import os
import numpy as np
import torch
import shutil
import sys
import logging
import SimpleITK as sitk
#from torch.utils.tensorboard import SummaryWriter
#import matplotlib.pyplot as plt
import random

def dataset_TVTsplit(inputs, targets, train_inds, val_inds, test_inds):
    def select_data(im_inds):
        selected_im = [inputs[ind] for ind in im_inds]
        selected_masks = [targets[ind] for ind in im_inds]
        return selected_im, selected_masks

    train_inputs, train_masks = select_data(train_inds)
    val_inputs, val_masks = select_data(val_inds)
    test_inputs, test_masks = select_data(test_inds)

    return train_inputs, train_masks, val_inputs, val_masks, test_inputs, test_masks

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def getDirs(parent_dir):
    ls = []
    for dir_name in os.listdir(parent_dir):
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path):
            ls.append(dir_name)
    return ls

def windowLevelNormalize(image, level, window):
    minval = level - window/2
    maxval = level + window/2
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld /= window
    return wld

def split_train_val(training_images, runNum, imagedir, maskdir):
    available_ims = getFiles(imagedir)
    available_masks = getFiles(maskdir)
    np.random.seed(training_images + (runNum * 17))
    choice = np.random.choice(range(len(available_ims)), size=(training_images+10), replace=False)
    train_ims = []
    train_masks = []
    val_ims = []
    val_masks = []
    count = 0
    for i in choice:
        if count < training_images:
            train_ims.append(available_ims[i])
            train_masks.append(available_masks[i])
            count += 1
        else:
            val_ims.append(available_ims[i])
            val_masks.append(available_masks[i])
    return train_ims, train_masks, val_ims, val_masks

def test_candidate(candidate, num_ims):
    if candidate < num_ims:
        return candidate
    return test_candidate(candidate-num_ims, num_ims)

def split_train_val_fold(num_training_images, foldNum, imagedir):
    available_ims = getFiles(imagedir)
    # shuffle from the alphanumeric order
    random.seed(2305)
    random.shuffle(available_ims)
    # determine images in this fold
    num_val_images=10
    if num_training_images < 10:
        num_val_images=5
    selection = []
    for image_num in range(num_training_images+num_val_images):
        candidate = (foldNum-1)*(num_training_images) + image_num
        selection.append(test_candidate(candidate, len(available_ims)))
    # allocate the fnames
    train_fnames, val_fnames = [], []   
    for i, idx in enumerate(selection):
        if i < num_training_images:
            train_fnames.append(available_ims[idx])
        else:
            val_fnames.append(available_ims[idx])
    return train_fnames, val_fnames

def selection_set(runNum, imagedir):
    available_ims = getFiles(imagedir)
    # shuffle from the alphanumeric order
    random.seed(2305)
    random.shuffle(available_ims)
    # determine selection images in this fold
    indices = []
    for image_num in range(30):
        candidate = (runNum+1)*10 + image_num
        indices.append(test_candidate(candidate, len(available_ims)))
    # allocate the fnames
    selection_fnames = []  
    for i, idx in enumerate(indices):
        selection_fnames.append(available_ims[idx])
    return selection_fnames

def save_itk(image, filename, spacing=[1,1,2]):
    image = image.astype(np.float32)
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetOrigin([0,0,0])
    itkimage.SetSpacing(spacing)
    sitk.WriteImage(itkimage, filename, True)

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

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
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)
    # old - save all
    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')