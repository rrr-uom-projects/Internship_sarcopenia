#sarcopenia_model_train
#update: 07/07/2021

#!pip install segmentation-models-pytorch
#!pip install torchsummary
#!pip install pytorchtools
#!pip install pytorch_toolbelt
#pip install torch.utils.tensorboard
#pip install tensorflow

#imports
from __future__ import division
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

from muscleMapper_utils import preprocess, splitandstick, k_fold_cross_val, dataset_TVTsplit, transforms
from muscleMapper_utils import weight, H_custom_Dataset, train, validate, test, get_lr
from muscleMapper_utils import getArea, getDensity, diceCoeff

#paths
save_path = "/home/hermione/Documents/Internship_sarcopenia/" 
data_path = "/home/hermione/Documents/Internship_sarcopenia/total_abstract_training_data 1.npz"

#%%
#loading the data
data = np.load(data_path, allow_pickle=True)
print([*data.keys()])
slices= data['slices']
masks = data['masks']
ids = data['ids']
masks_slb = data['boneless']
bone_masks = data['bone_masks']
pixel_area = data['areas']

slices_processed, masks_processed = preprocess(slices, masks_slb)

#split into training and testing
print("Lengths: ", len(masks), len(slices), len(slices_processed), len(masks_processed))
dataset_size = len(masks)
train_array, test_array = k_fold_cross_val(dataset_size, num_splits = 5)

for i in range(0, 5):
  #split_val = test_array[i]
  val_split, train_split = np.split(train_array[i], [7], axis= 0)#[16]
  ids_test = ids[(test_array[i])]
  ids_val = ids[val_split]
  ids_train = ids[train_split]
  
  slice_train, masks_train, slice_val, masks_val, slice_test, masks_test = dataset_TVTsplit(slices_processed, masks_processed, train_split, val_split, test_array[i])

  slice_train, slice_val, slice_test = splitandstick(slices_processed)
  masks_train, masks_val, masks_test = splitandstick(masks_processed)
  ids_train, ids_val, ids_test = splitandstick(ids)
  bone_masks_train, bone_masks_val, bone_masks_test = splitandstick(bone_masks)

  print(slice_test.shape, slice_train.shape)
  #print(ids_test[3])
  plt.imshow(slice_test[0],cmap = "gray")
  #masks_test[0][masks_test[0]==0] = np.nan
  plt.imshow(masks_test[0], cmap = "cool", alpha = 0.5)
  fig = plt.figure(figsize=(20, 10))
  ax=[]
  for i in range(0,10):
    print(ids_test[i])
    ax.append(fig.add_subplot(2,5, i+1))
    plt.imshow(slice_test[i])
    plt.axis("off")
  plt.show()

  #%%
  #classs inbalence
  #ratio of no of 1s over no of 0s. averaged
  weight = weight(masks_train)

  #transform the data
  train_transform = A.Compose([
    A.Resize(260, 260),
    A.RandomSizedCrop(min_max_height=(200, 260), height=260, width=260, p=0.2),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.8, alpha_affine=120 * 0.05, p= 0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensor()
  ])

  val_transform = A.Compose(
      [A.Resize(260, 260), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensor()]
  )

  test_transform = A.Compose(
      [A.Resize(260, 260), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensor()]
  )

  train_dataset = H_custom_Dataset(slice_train, masks_train, transform = train_transform)#transforms = aug
  test_dataset = H_custom_Dataset(slice_test, masks_test, transform = test_transform)
  val_dataset = H_custom_Dataset(slice_val, masks_val, transform = val_transform)

  train_dataloader = DataLoader(train_dataset, batch_size = 8, num_workers = 2, shuffle = True)#used batch size 4 for sem1
  test_dataloader = DataLoader(test_dataset, batch_size = len(slice_test), num_workers = 2, shuffle = False)
  val_dataloader = DataLoader(val_dataset, batch_size = 8, num_workers = 2, shuffle = True)

  #set up tensorboard
  writer = SummaryWriter()

  # Load pre-trained model
  pt_model = models.segmentation.fcn_resnet50(pretrained=True)
  model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
  # Change final layer as pre-trained model expects 21 outputs
  pt_dict = pt_model.state_dict()
  model_dict = model.state_dict()
  pretrained_dict = {}
  for key, val in pt_dict.items():
      if key in model_dict:
          if val.shape == model_dict[key].shape:
              pretrained_dict[key] = val
          else:
              print("Shapes don't match")
              continue
      else:
          print("key not in dict")
          continue
  # Overwrite entries in the existing state dict
  model_dict.update(pretrained_dict)
  # Load new state dict
  model.load_state_dict(model_dict)
  #print(model)

  #hyperparameters
  loss = L.BinaryFocalLoss()
  #loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight).cuda()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = loss
  scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
  #scheduler = lr_scheduler.OneCycleLR(optimizer, step_size=25, gamma=0.5)

  #%%
  train_loss , train_accuracy = [], []
  val_loss , val_accuracy = [], []
  start = time.time()
  num_epochs = 10
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

  #training the model
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print("\nlearning rate: ", get_lr(optimizer))
    train_epoch_loss = train(model, train_dataloader, device, optimizer, criterion, epoch, writer)
    val_epoch_loss = validate(model, val_dataloader, device, criterion, epoch, writer)
    train_loss.append(train_epoch_loss)
    print("train loss: ", train_epoch_loss) 
    val_loss.append(val_epoch_loss)
    print("val loss: ", val_epoch_loss) 
    scheduler.step()

  print("train loss: ", train_loss)
  print("val loss: ", val_loss)
  end = time.time()
  print((end-start)/60, 'minutes')
  #%%
  #save model weights
  model_stat_dict_300_FL = torch.save(model.cpu().state_dict(), "/home/hermione/Documents/Internship_sarcopenia/model_state_dict.pt")
  #save the loss
  loss = {'Training': train_loss, 'Validation': val_loss}
  loss_table = np.transpose(np.array(loss))
  print(loss)
  #loss_table = np.savetxt("/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/loss_08_07.csv", loss, delimiter=',')
  l_df = pd.DataFrame(loss)
  l_df.to_excel(excel_writer = "/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/loss.xlsx")

  #%%
  #testing the model
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")
  model.to(device)
  c3s, test_predictions = test(model, test_dataloader)
  print(test_predictions.shape, c3s.shape)
  test_predictions = torch.from_numpy(test_predictions)
  #removing the bone masks from the models predictions
  segment_pred_slb = np.logical_and(test_predictions, bone_masks_test[:,np.newaxis,...])
  segment_pred_slb = (segment_pred_slb.float())#.numpy()
  print(np.unique(segment_pred_slb))

  fig=plt.figure(figsize=(20, 10))
  ax = []
  rows = 2
  columns = 5
  for i in range(0, len(c3s)):
    ax.append(fig.add_subplot(rows,columns, i+1))
    #segment_pred_slb[i][segment_pred_slb[i]==0] = np.nan #uncomment to get pretty images
    plt.imshow(c3s[i,0,...], cmap="gray")
    plt.imshow(segment_pred_slb[i,0,...], cmap = "autumn", alpha = 0.5)
    ax[-1].set_title("Network test:"+str(i))
    plt.axis("off")
  #plt.show()
  #%%
  #Dice - comparing our netowrks output to the GTs
  for batch_idx, test_dataset in enumerate(test_dataloader):  
        test_em, test_lab = test_dataset[0], test_dataset[1]
        break
  dice_net_v_pred = diceCoeff(segment_pred_slb, test_lab, smooth=1, activation=None)

  print("Dice ", dice_net_v_pred)
  mean = np.mean(dice_net_v_pred.item())
  print("Dice: ", mean)

#Dice scores box plot of the different test folds here.


# #Area and Density of SM in the tests
# ct_scans = np.array(slice_test)
# #network_pred = segment_pred_slb[:,0,...]

# print("Patients IDs in test data: ", ids_test)
# ids_test = np.array(ids_test)
# test_index = []
# for i in range(0, len(ids)):
#   if any(ids[i] == j for j in ids_test) == True:
#     test_index.append(i)
#     print(i)
# print(test_index)
# #%%
# pixel_area_id = pixel_area[test_index]
# pixel_area_id = np.array(pixel_area_id*(0.1*0.1))#cm^2->mm^2
# print("pixel areas of the test images: ", pixel_area_id)
# #pixel_area = np.repeat(np.array(pixel_area)*(0.1*0.1), 2)
# #print(pixel_area.shape)
# #%%

# ## Note that the areas have some thresholds applied that are from the literature
# extractionDict = {"sma" : partial(getArea, thresholds=(-30, +130)), 
# 				  "smd" : partial(getDensity)
# 				 }
# feat_list = ["sma","smd"]
# # Extract features from slices_processed
# feature_list_net = []
# feature_list_ours = []
# for i in range(0,len(ct_scans)):
#   feature_list_net.append([extractionDict[a](ct_scans[i], segment_pred_slb[i], pixel_area_id[i]) for a in feat_list])
#   #feature_list_ours.append([extractionDict[a](ct_scans[i], ground_truths[i], pixel_area[i]) for a in feat_list])
  
# print("Area, Density: ", feature_list_net)

# #for the network - avergae sma and smd
# sma = np.array(feature_list_net)[:,0]
# mean_area = np.mean(sma)
# area_sd = ndimage.standard_deviation(sma)
# print(mean_area, "cm^2 ", "sd: ", area_sd)

# smd = np.array(feature_list_net)[:,1]
# mean_density = np.mean(smd)
# den_sd = ndimage.standard_deviation(smd)
# print(mean_density, "HU" ,"sd", den_sd)

# #saving to an excel file
# array = np.transpose(np.array(feature_list_net))
# print(array)
# df = pd.DataFrame(array, index= feat_list, columns=ids).T
# #df.to_excel(excel_writer = "/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/muscle_area_and_density_training_data_07_07.xlsx")
