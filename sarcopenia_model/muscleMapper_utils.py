
import random
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import math
from scipy import ndimage
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses as L
from PIL import Image
import time
import copy
from tqdm import tqdm
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import albumentations as A
import random
from albumentations.pytorch import ToTensor
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import pandas as pd
from decimal import Decimal

def k_fold_cross_val(dataset_size, num_splits):
    train =[]
    test = []
    np.random.seed(2305)
    #shuffled_ind_list = np.random.permutation(dataset_size)
    shuffled_ind_list = np.arange(dataset_size)
    #print(shuffled_ind_list)
    kf = KFold(n_splits = num_splits, shuffle = True, random_state=np.random.seed(2305))
    for train_index, test_index in kf.split(shuffled_ind_list):
          print("TRAIN:", train_index, "\nTEST:", test_index)
          train.append(train_index)
          test.append(test_index)
    return train, test

def dataset_TVTsplit(inputs, targets, bone_msks, train_inds, val_inds, test_inds):
    def select_data(im_inds):
        selected_im = [inputs[ind] for ind in im_inds]
        selected_masks = [targets[ind] for ind in im_inds]
        selected_bone_msks = [bone_msks[ind] for ind in im_inds]
        return np.array(selected_im), np.array(selected_masks), np.array(selected_bone_msks)

    train_inputs, train_masks, train_bone = select_data(train_inds)
    val_inputs, val_masks, val_bone = select_data(val_inds)
    test_inputs, test_masks, test_bone = select_data(test_inds)

    return train_inputs, train_masks, val_inputs, val_masks, test_inputs, test_masks, test_bone

#definitions
def diceCoeff(pred, gt, smooth=1, activation='sigmoid'):
    """ computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")
 
    pred = np.round(activation_fn(pred))
    N = gt.size(0) #n should be batch size
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    print(intersection, unionset)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N

def preprocess(slice_array, masks_array):
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  window = 350
  level = 50
  size = len(masks_array)
  vmin = (level/2) - window
  vmax = (level/2) + window
  processed_slices = []
  processed_masks = (np.nan_to_num(masks_array)).astype(np.float64)
  for i in range (0, size):
    slice_array[i][slice_array[i]>vmax] = vmax
    slice_array[i][slice_array[i]<vmin] = vmin
    shape = slice_array[i].shape
    image_scaled = np.round(sklearn.preprocessing.minmax_scale(slice_array[i].ravel(), feature_range=(0,1)), decimals = 10).reshape(shape)
    processed_slices.append(image_scaled)
    processed_masks[i][processed_masks[i]>1] = 1
  processed_slices = np.array(processed_slices)
  return processed_slices, processed_masks

def splitandstick(array):
  #a,b,c,aa,bb,cc = np.split(array,[25,30,35,60,65])
  b,c,a,bb,cc,aa = np.split(array,[5,10,35,40,45])
  a = np.concatenate((a,aa))
  b = np.concatenate((b,bb))
  c = np.concatenate((c,cc))
  return a,b,c

def genTransforms(slice_array, masks_array): #getting axis in the require places and applying filters to the image
  size = len(masks_array)
  slices_3chan = []
  if slice_array.shape != (size,3,...):
    slices_3chan = np.repeat(slice_array[:,:,:,np.newaxis], 3, axis=-1)
    for i in range (0,len(masks_array)):
      slices_3chan[i,:,:,1] = ndimage.gaussian_laplace(slices_3chan[i,:,:,1], sigma=1)
      slices_3chan[i,:,:,2] = ndimage.sobel(slices_3chan[i,:,:,2])
  transform_slice = slices_3chan.astype(np.float32)
  transform_mask = masks_array.astype(np.float32)
  return transform_slice, transform_mask

#classes
class H_custom_Dataset(TensorDataset):
    def __init__(self, images, masks, transform=None):
        super(H_custom_Dataset, self).__init__()
        self.transform = transform
        self.images, self.masks  = genTransforms(images, masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
      if self.transform:
        augmentations = self.transform(image=self.images[idx], mask=self.masks[idx])
        image = augmentations["image"]
        mask =augmentations["mask"]
      else:
        image = self.images
        mask = self.masks
      return image, mask
  
# training function
def train(model, train_dataloader, device, optimizer, criterion, epoch, writer):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    total_step = len(train_dataloader)
    for batch_idx, train_dataset in enumerate(train_dataloader):
        slice_train, masks_train = train_dataset[0].to(device), train_dataset[1].to(device)
        slice_train = slice_train.type(torch.float32)
        optimizer.zero_grad()
        output = model(slice_train)
        #print(slice_train.shape, output["out"].shape, masks_train.shape)
        loss = criterion(output["out"], masks_train)
        train_running_loss += loss.item()
        _, preds = torch.max(output["out"], 1)
        train_running_correct += (preds == masks_train).sum().item()
        loss.backward()
        optimizer.step()  
    train_loss = train_running_loss/len(train_dataloader.dataset)
    #train_accuracy = train_running_correct/len(train_dataloader.dataset)
    #train_accuracy = diceCoeff(output["out"], masks_train)
    writer.add_scalar('training loss', train_loss, epoch)
    #print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    #writer.add_scalar('training accuracy', train_accuracy, epoch)
    return train_loss

def validate(model, val_dataloader, device, criterion, epoch, writer):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    torch.no_grad()
    for int, data in enumerate(val_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        data = data.type(torch.float32)
        #target.cuda()
        output = model(data)
        #print("output shape: ", output["out"].shape)
        loss = criterion(output["out"], target)
        
        val_running_loss += loss.item()
        _, preds = torch.max(output["out"], 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss/len(val_dataloader.dataset)
    #writing to tensorboard
    writer.add_scalar('Validation loss', val_loss, epoch)
    #val_accuracy = val_running_correct/len(val_dataloader.dataset)
    #val_accuracy = diceCoeff(output["out"], target)
    #writer.add_scalar('Validation accuracy', val_accuracy, epoch)
    return val_loss

def test(model, test_dataloader, device):
    model.eval()
    segments = []
    c3s = []
    sigmoids = []

    for int, data in enumerate(test_dataloader):
        slices_test = data[0].to(device)
        slices_test = slices_test.type(torch.float32)
        output = model(slices_test)["out"]
        print("output shape: ", output.shape)
        print(torch.max(output), torch.min(output))
        test_ouput = output.detach().cpu()
        slices_test = slices_test.detach().cpu()
        sigmoid = 1/(1 + np.exp(-test_ouput))
        segment = (sigmoid > 0.5).float()
        print(np.unique(segment))

        if int == 0:
            segments = segment
            c3s = slices_test
            sigmoids = sigmoid.float()
        else:
            segments.append(np.array(segment))
            c3s.append(np.array(slices_test))
            sigmoids.append(np.array(sigmoid.float()))
  
    return np.array(c3s), np.array(segments), np.array(sigmoids)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
#functions to calculate SM area and density from the segmentations
def getArea(image, mask, area, label=1, thresholds = None):
  sMasks = (mask == label)
  threshold = np.logical_and(image > (thresholds[0]), image <  (thresholds[1]))
  tmask = np.logical_and(sMasks, threshold)
  return np.count_nonzero(tmask) * area

def getDensity(image, mask, area, label=1):#pixel density
  if image.shape != (len(image),1,...):
    mask = np.squeeze(mask)
  return float(np.mean(image[np.where(mask == 1)]))#-1024

def weight(masks):
    ratio = []
    for i in range(len(masks)):
      no_of_0s = (masks[i] == 0).sum()
      no_of_1s = (masks[i] == 1).sum()
      #print(no_of_1s, no_of_0s)
      ratio.append(no_of_1s/no_of_0s)

    average_ratio = np.mean(ratio)
    print(average_ratio)
    weights = (1/average_ratio) #*10 to penalise the network more harshly for getting it wrong
    weight = torch.Tensor([weights])
    print(weight)
    return weight

def do_it_urself_round(number, decimals = 0, is_array = False):
  float_num = number
  dec_num =  Decimal(str(number))
  Ddecimal = Decimal(str(decimals))
  round_num = round(dec_num, decimals)
  diff = np.abs(dec_num - round_num)
  round_diff = Decimal(str(0.5/(10**decimals)))
  if (diff == round_diff and dec_num >= round_num):
    round_num += (1/(10**(Ddecimal))) 
  if decimals == 0:
    return int(round_num)
  else:
    return float(round_num)