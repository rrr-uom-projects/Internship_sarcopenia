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

import tensorflow as tf
tf.test.gpu_device_name()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_VISIBLE_DEVICES']='2, 3'

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
def train(model, train_dataloader, device):
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
    #writer.add_scalar('training loss', train_loss, epoch)
    #print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    #writer.add_scalar('training accuracy', train_accuracy, epoch)
    return train_loss

def validate(model, val_dataloader, device):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
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
    #writer.add_scalar('Validation loss', val_loss, epoch)
    #val_accuracy = val_running_correct/len(val_dataloader.dataset)
    #val_accuracy = diceCoeff(output["out"], target)
    #writer.add_scalar('Validation accuracy', val_accuracy, epoch)
    return val_loss

def test(model, test_dataloader):
    model.eval()
    segments = []
    c3s = []

    for int, data in enumerate(test_dataloader):
        slices_test = data[0].to(device)
        slices_test = slices_test.type(torch.float32)
        output = model(slices_test)["out"]
        print("output shape: ", output.shape)
        test_ouput = output.detach().cpu()
        slices_test = slices_test.detach().cpu()
        sigmoid = 1/(1 + np.exp(-test_ouput))
        segment = (sigmoid > 0.5).float()
        print(np.unique(segment))
        #print(int)
        if int == 0:
          segments = segment
          c3s = slices_test
        else:
          segments = np.append(segments, np.array(segment), axis = 0)
          c3s = np.append(c3s, np.array(slices_test), axis = 0)
  
    segments = np.array(segments)
    c3s = np.array(c3s)
    return c3s, segments

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

#%%
#loading the data
data_path = "/home/hermione/Documents/Internship_sarcopenia/total_abstract_training_data 1.npz"
#data_path = "C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\Mphys sem 2\\training data\\total_abstract_training_data.npz"
data = np.load(data_path, allow_pickle=True)
print([*data.keys()])
slices= data['slices']
masks = data['masks']
ids = data['ids']
masks_slb = data['boneless']
bone_masks = data['bone_masks']
pixel_areas = data['areas']

slices_processed, masks_processed = preprocess(slices, masks_slb)

#split into training and testing
"""slice_train, slice_test, masks_train, masks_test, ids_train, ids_test = train_test_split(slices_processed, masks_processed, ids, test_size=(10/35), random_state = 5)
slice_test, slice_val, masks_test, masks_val, ids_test, ids_val = train_test_split(slice_test, masks_test, ids_test, test_size = 0.5, random_state = 5 )

print(slice_train.shape)
"""
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


#%%
#classs inbalence
#ratio of no of 1s over no of 0s. averaged
ratio = []
for i in range(len(masks_train)):
  no_of_0s = (masks_train[i] == 0).sum()
  no_of_1s = (masks_train[i] == 1).sum()
  #print(no_of_1s, no_of_0s)
  ratio.append(no_of_1s/no_of_0s)

average_ratio = np.mean(ratio)
print(average_ratio)
weights = (1/average_ratio) #*10 to penalise the network more harshly for getting it wrong
weight = torch.Tensor([weights])
print(weight)

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
#writer = SummaryWriter()

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
start = time.time()
num_epochs = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#training the model
for epoch in range(num_epochs):
  print('Epoch {}/{}'.format(epoch+1, num_epochs))
  print("\nlearning rate: ", get_lr(optimizer))
  train_epoch_loss = train(model, train_dataloader, device)
  val_epoch_loss = validate(model, val_dataloader, device)
  train_loss.append(train_epoch_loss)
  #train_accuracy.append(train_epoch_accuracy)
  print("train loss: ", train_epoch_loss) 
  #print() "train acc: ", train_accuracy)
  val_loss.append(val_epoch_loss)
  #val_accuracy.append(val_epoch_accuracy)
  print("val loss: ", val_epoch_loss) 
  #print("val acc: ", val_accuracy)
  scheduler.step()

print("train loss: ", train_loss)
print("val loss: ", val_loss)
end = time.time()
print((end-start)/60, 'minutes')

#save model weights
model_stat_dict_300_FL = torch.save(model.cpu().state_dict(), "/content/model_state_dict_300_FL.pt")
#save the loss
loss = np.concatenate((np.asarray(train_loss), np.asarray(val_loss)), axis = 0)
print(loss)
loss = np.savetxt("/content/loss_16_02.csv", loss, delimiter=',')

#%%
#testing the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
c3s, test_predictions = test(model, test_dataloader)
print(test_predictions.shape, c3s.shape)
test_predictions = torch.from_numpy(test_predictions)
#removing the bone masks from the models predictions
segment_pred_slb = np.logical_and(test_predictions, bone_masks[:,np.newaxis,...])
segment_pred_slb = (segment_pred_slb.float()).numpy()
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
plt.show()
#%%
#Dice - comparing our netowrks output to the GTs
#dice_array = []
for batch_idx, test_dataset in enumerate(test_dataloader):  
      test_em, test_lab = test_dataset[0], test_dataset[1]
      break
dice_net_v_pred = diceCoeff(segment_pred_slb, test_lab, smooth=1, activation=None)

print("Dice ", dice_net_v_pred)
mean = np.mean(dice_net_v_pred.item())
print("Dice: ", mean)

#Area and Density of SM in the tests
ct_scans = np.array(slice_test)
network_pred = segment_pred_slb[:,0,...]

print("Patients IDs in test data: ", ids_test)
#%%
pixel_area_id = [pixel_area[33-3],pixel_area[34-3],pixel_area[35-4],pixel_area[36-3],pixel_area[38-4],pixel_area[33-3],pixel_area[34-3],pixel_area[35-4],pixel_area[36-3],pixel_area[38-4]]
pixel_area_id = np.array(pixel_area_id)
print(pixel_area_id)
pixel_area = np.repeat(np.array(areas)*(0.1*0.1), 2)
print(pixel_area.shape)
#%%

## Note that the areas have some thresholds applied that are from the literature
extractionDict = {"sma" : partial(getArea, thresholds=(-30, +130)), 
				  "smd" : partial(getDensity)
				 }
feat_list = ["sma","smd"]
# Extract features from slices_processed
feature_list_net = []
feature_list_ours = []
for i in range(0,len(ct_scans)):
  feature_list_net.append([extractionDict[a](ct_scans[i], segment_pred_slb[i], pixel_area_id[i]) for a in feat_list])
  #feature_list_ours.append([extractionDict[a](ct_scans[i], ground_truths[i], pixel_area[i]) for a in feat_list])
  
print("Area, Density: ", feature_list_net)

#for the network - avergae sma and smd
sma = np.array(feature_list_net)[:,0]
mean_area = np.mean(sma)
area_sd = ndimage.standard_deviation(sma)
print(mean_area, "cm^2 ", "sd: ", area_sd)

smd = np.array(feature_list_net)[:,1]
mean_density = np.mean(smd)
den_sd = ndimage.standard_deviation(smd)
print(mean_density, "HU" ,"sd", den_sd)

#saving to an excel file
array = np.transpose(np.array(feature_list_ours))
print(array)
df = pd.DataFrame(array, index= feat_list, columns=ids).T
df.to_excel(excel_writer = "/content/muscle_area_and_density_training_data.xlsx")
