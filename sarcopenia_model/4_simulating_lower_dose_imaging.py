#simulating lower dose imaging through augmentations.
#test how well the model generalises through an ablation table of the dice scores of the ten extra patients.
#created: 30/11/2021

#imports
from __future__ import division
import cv2
from albumentations.augmentations.transforms import Downscale, GaussNoise, GaussianBlur
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import math
from pytorch_toolbelt.losses import dice
from scipy import ndimage
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

from muscleMapper_utils import preprocess, do_it_urself_round, splitandstick, k_fold_cross_val, dataset_TVTsplit, transforms
from muscleMapper_utils import weight, H_custom_Dataset, train, validate, test, get_lr
from muscleMapper_utils import getArea, getDensity, diceCoeff

#paths
save_path = "/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/" 
data_path = "/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/total_abstract_training_data 1.npz"

#%%
#loading the data
data = np.load(data_path, allow_pickle=True)
print([*data.keys()])
slices, slices2 = np.split(data['slices'],2)
masks, masks2 = np.split(data['masks'],2)
ids, ids2 = np.split(data['ids'],2)
masks_slb, masks_slb2 = np.split(data['boneless'],2)
bone_masks, bone_masks2 = np.split(data['bone_masks'],2)
pixel_area, pixel_area2 = np.split(data['areas'],2)

#the ten extra slices can comment out if not training with them.
extra_data = '/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/save_extras.npz'
extra_data = np.load(extra_data, allow_pickle=True)
Eslices = extra_data['slices']
Emasks = extra_data['masks']
Ebone = extra_data['bone']
Emasks_slb = extra_data['masks_slb']
Eids = extra_data['ids']
Eareas = extra_data['areas']

slices_processed, masks_processed = preprocess(slices, masks_slb)
slices_processed2, masks_processed2 = preprocess(slices2, masks_slb2)
Eslices_processed, Emasks_processed = preprocess(Eslices, Emasks)

#split into training and testing  #35 7 42 6 folds
fold_num = 5
print("Lengths: ", len(masks), len(slices), len(slices_processed), len(masks_processed))
dataset_size = len(masks)
train_array, test_array = k_fold_cross_val(dataset_size, num_splits = fold_num)

test_dice_scores = []

for i in range(fold_num):

  #make fold files to save info
  save_dir = save_path + "ablation_MM_fold" + str(i+1)
  try:
      os.makedirs(save_dir)
  except OSError: #if already exists
      pass

  #split into train and val
  val_split, train_split = np.split(train_array[i], [5], axis = 0)#5,25
  ids_test = np.concatenate((ids[(test_array[i])], ids2[(test_array[i])]))
  ids_val = np.concatenate((ids[val_split], ids2[val_split]))
  ids_train = np.concatenate((ids[train_split], ids2[train_split]))
  
  slice_train, masks_train, slice_val, masks_val, slice_test, masks_test, bone_masks_test = dataset_TVTsplit(slices_processed, masks_processed, bone_masks, train_split, val_split, test_array[i])
  slice_train2, masks_train2, slice_val2, masks_val2, slice_test2, masks_test2, bone_masks_test2 = dataset_TVTsplit(slices_processed2, masks_processed2, bone_masks2, train_split, val_split, test_array[i])
   
  slice_train = np.concatenate((slice_train, slice_train2))
  masks_train = np.concatenate((masks_train, masks_train2))
  slice_val = np.concatenate((slice_val, slice_val2))
  masks_val = np.concatenate((masks_val, masks_val2))
  slice_test = np.concatenate((slice_test, slice_test2))
  masks_test = np.concatenate((masks_test, masks_test2))
  bone_masks_test = np.concatenate((bone_masks_test, bone_masks_test2))
  
  print(slice_test.shape, slice_val.shape, slice_train.shape)
  #%%
  #classs inbalence
  #ratio of no of 1s over no of 0s. averaged
  #weight = weight(masks_train)

  #transform the data
  #training augmentations
  #augmentations to simulate lower dose imaging
  ldi_augs = [Downscale(0.25,0.25), GaussNoise(var_limit=0.005), GaussianBlur(blur_limit=1)]
  
  train_transform = A.Compose([
    A.Resize(260, 260),
    A.RandomSizedCrop(min_max_height=(200, 260), height=260, width=260, p=0.2),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.8, alpha_affine=120 * 0.05, p= 0.2),
    A.OneOf([Downscale(0.25,0.25), GaussNoise(var_limit=0.005), GaussianBlur(blur_limit=1)],p=1), #to reduce image quality could use noise or dropout
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
  writer = SummaryWriter(log_dir=save_dir)

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

  #%%
  train_loss , train_accuracy = [], []
  val_loss , val_accuracy = [], []
  running_average = []
  patience = 0
  start = time.time()
  num_epochs = 300
  device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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

  #print("train loss: ", train_loss)
  #print("val loss: ", val_loss)
  end = time.time()
  print((end-start)/60, 'minutes')
  #%%
  #save model weights
  model_path = save_dir + "/model_state_dict.pt"
  model_stat_dict_300_FL = torch.save(model.cpu().state_dict(), model_path)
  #save the loss
  loss = {'Training': train_loss, 'Validation': val_loss}
  #loss_table = np.transpose(np.array(loss))
  print(loss)
  #loss_table = np.savetxt("/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/loss_08_07.csv", loss, delimiter=',')
  l_df = pd.DataFrame(loss)
  l_df.to_excel(excel_writer = save_dir + "/loss.xlsx", index=False, sheet_name=f'fold{i+1}')

  #%%
  #testing the model
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")
  model.to(device)
  c3s, test_predictions, sig = test(model, test_dataloader,device)
  print(test_predictions.shape, c3s.shape)
  test_predictions = torch.from_numpy(test_predictions)
  #removing the bone masks from the models predictions
  segment_pred_slb = np.logical_and(test_predictions, bone_masks_test[:,np.newaxis,...])
  segment_pred_slb = segment_pred_slb.float()
  #segment_pred_slb = segment_pred_slb.astype(float)
  print("segment unique values", np.unique(segment_pred_slb))

  #%%
  #Dice - comparing our netowrks output to the GTs
  for batch_idx, test_dataset in enumerate(test_dataloader):  
        test_em, test_lab = test_dataset[0], test_dataset[1]
        break
  dice_net_v_pred = []
  for i in range(len(segment_pred_slb)):
    dice_net_v_pred.append(diceCoeff(segment_pred_slb[i], test_lab[i], smooth=1, activation=None))

  print("Dice ", dice_net_v_pred)
  mean = np.mean(dice_net_v_pred)
  print("Dice mean: ", mean)
  test_dice_scores.append(dice_net_v_pred)

  #display the test images
  fig=plt.figure(figsize=(20, 10))
  ax = []
  rows = 3
  columns = 5
  for i in range(0, len(c3s)):
    ax.append(fig.add_subplot(rows,columns, i+1))
    segment_pred_slb[i][segment_pred_slb[i]==0] = np.nan #uncomment to get pretty images
    plt.imshow(c3s[i,0,...], cmap="gray")
    plt.imshow(segment_pred_slb[i,0,...], cmap = "autumn", alpha = 0.5)
    #plt.imshow(sig[i,0,...], cmap = "cool", alpha = 0.5)
    ax[-1].set_title("Network test:"+str(i))
    plt.axis("off")
  plt.savefig(save_dir + "/MME_test.png")
  plt.close()
  #plt.show()

#Dice scores box plot of the different test folds here.
cols = []
dict = {}
median = []
test_dice_scores = np.array(test_dice_scores)
print(test_dice_scores.shape)
for j in range(fold_num):
    cols.append(f'fold{j+1}')
    dict[f'fold{j+1}'] = test_dice_scores[j]
    median.append(do_it_urself_round(np.nanmedian(test_dice_scores[j]),2))
    print(test_dice_scores[j].shape)
dfbp = pd.DataFrame(dict)
plt.figure()
plt.boxplot(dfbp.dropna().values, 0,  'r')
plt.ylabel("Test Dice Scores")
plt.grid(True, linestyle='-', which='major', color='lightgrey',
            alpha=0.5)
for i in range(len(median)):
    plt.text(i+0.75, median[i]-0.005, median[i])
plt.savefig("MME_fold_info/box_plot.png")
print("Saved Test Info.")

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
