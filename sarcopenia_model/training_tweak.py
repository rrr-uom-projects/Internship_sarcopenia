#file to retrain the MM model with ten extra peices of data.

#imports
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import math
from pytorch_toolbelt.losses import dice
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

from muscleMapper_utils import preprocess, do_it_urself_round, k_fold_cross_val, dataset_TVTsplit, transforms
from muscleMapper_utils import H_custom_Dataset, train, validate, test, get_lr
from muscleMapper_utils import diceCoeff

#paths
save_path = "/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/" 
extra_data = '/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/save_extras.npz'
MM_model_weights_path = "/home/hermione/Documents/Internship_sarcopenia/Inference/MM3_model_state_dict_fold6.pt"

#%%
#loading the data
extra_data = np.load(extra_data, allow_pickle=True)
Eslices = extra_data['slices']
Emasks = extra_data['masks']
Ebone = extra_data['bone']
Emasks_slb = extra_data['masks_slb']
Eids = extra_data['ids']
Eareas = extra_data['areas']

Eslices_processed, Emasks_processed = preprocess(Eslices, Emasks,level = 50, window =500)

#split into training and testing  #2 2 6 5 folds
fold_num = 5
print(len(Eslices_processed), len(Emasks_processed))
Etrain_arr, Etest_arr = k_fold_cross_val(len(Emasks_processed), num_splits=fold_num)

test_dice_scores = []

for i in range(fold_num):

  #make fold files to save info
  save_dir = save_path + "MM_tweak2_fold" + str(i+1)
  try:
      os.makedirs(save_dir)
  except OSError: #if already exists
      pass

  #split into train and val
  Eval_split, Etrain_split = np.split(Etrain_arr[i],[2], axis = 0)#2,6
  test_split_train, Etest_arr[i] = np.split(Etest_arr[i],[1], axis = 0)
  Etrain_split = np.concatenate((Etrain_split, test_split_train))
  slice_train, masks_train, slice_val, masks_val, slice_test, masks_test, bone_masks_test = dataset_TVTsplit(Eslices_processed, Emasks_processed, Ebone, Etrain_split, Eval_split, Etest_arr[i])
  
  print(slice_train.shape, masks_train.shape)
  print(slice_val.shape, masks_val.shape)
  print(slice_test.shape, masks_test.shape)

  #transform the data
  train_transform = A.Compose([
    A.Resize(240, 240),
    A.RandomSizedCrop(min_max_height=(200, 240), height=240, width=240, p=0.2),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.8, alpha_affine=120 * 0.05, p= 0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensor()
  ])

  val_transform = A.Compose(
      [A.Resize(240, 240), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensor()]
  )

  test_transform = A.Compose(
      [A.Resize(240, 240), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensor()]
  )

  train_dataset = H_custom_Dataset(slice_train, masks_train, transform = train_transform)#transforms = aug
  test_dataset = H_custom_Dataset(slice_test, masks_test, transform = test_transform)
  val_dataset = H_custom_Dataset(slice_val, masks_val, transform = val_transform)

  train_dataloader = DataLoader(train_dataset, batch_size = 4, num_workers = 2, shuffle = True)#used batch size 4 for sem1
  test_dataloader = DataLoader(test_dataset, batch_size = len(slice_test), num_workers = 2, shuffle = False)
  val_dataloader = DataLoader(val_dataset, batch_size = 4, num_workers = 2, shuffle = True)

  #set up tensorboard
  writer = SummaryWriter(log_dir=save_dir)

  # Load pre-trained model
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
  model.load_state_dict(torch.load(MM_model_weights_path, map_location=device))

  #hyperparameters
  loss = L.BinaryFocalLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.00001)
  criterion = loss
  scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

  #%%
  train_loss , train_accuracy = [], []
  val_loss , val_accuracy = [], []
  running_average = []
  patience = 0
  start = time.time()
  num_epochs = 40
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

  end = time.time()
  print((end-start)/60, 'minutes')
  #%%
  #save model weights
  model_path = save_dir + "/model_state_dict.pt"
  model_stat_dict_300_FL = torch.save(model.cpu().state_dict(), model_path)
  #save the loss
  loss = {'Training': train_loss, 'Validation': val_loss}
  print(loss)
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
  rows = 1
  columns = 2
  for i in range(0, len(c3s)):
    ax.append(fig.add_subplot(rows,columns, i+1))
    segment_pred_slb[i][segment_pred_slb[i]==0] = np.nan #uncomment to get pretty images
    plt.imshow(c3s[i,0,...], cmap="gray")
    plt.imshow(segment_pred_slb[i,0,...], cmap = "autumn", alpha = 0.5)
    ax[-1].set_title("Network test:"+str(i))
    plt.axis("off")
  plt.savefig(save_dir + "/MM_tweak_test.png")
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
plt.savefig(save_path+"MM_tweak2_fold_info/box_plot.png")
print("Saved Test Info.")
