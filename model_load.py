# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:41:29 2021

@author: Olivia

test change
"""
#%%

#imports
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sklearn
import albumentations as A
from albumentations.pytorch import ToTensor
from functools import partial
from torchvision import datasets, models, transforms
######import segmentation_models_pytorch as smp
import pandas as pd
from sklearn import preprocessing
#%%
# function definitions
# function to preprocess masks and slices
def preprocess(slice_array, masks_array):
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

# function to calculate the dice coefficient between two masks
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

# function to convert input arrays into three channel images with sobel and LoG fliters applied
def genTransforms(slice_array, masks_array):
    
  size = len(masks_array)
  slices_3chan = []
  
  if slice_array.shape != (size,3,...):
    slices_3chan = np.repeat(slice_array[:,:,:,np.newaxis], 3, axis=-1)
    # apply filters to two channels
    for i in range (0,len(masks_array)):
      slices_3chan[i,:,:,1] = ndimage.gaussian_laplace(slices_3chan[i,:,:,1], sigma=1)
      slices_3chan[i,:,:,2] = ndimage.sobel(slices_3chan[i,:,:,2])

  transform_slice = slices_3chan.astype(np.float32)
  transform_mask = masks_array.astype(np.float32)
  return transform_slice, transform_mask

# testing function
def test(model, test_dataloader):
    model.eval()
    segments = []
    c3s = []

    for int, data in enumerate(test_dataloader):
        slices_test = data[0].to(device)
        slices_test = slices_test.type(torch.float32)
        output = model(slices_test)["out"]
        
        test_ouput = output.detach().cpu()
        slices_test = slices_test.detach().cpu()
        # sigmoid and thresholding
        sigmoid = 1/(1 + np.exp(-test_ouput))
        segment = (sigmoid > 0.5).float()
       
        if int == 0:
          segments = segment
          c3s = slices_test
        else:
          segments = np.append(segments, np.array(segment), axis = 0)
          c3s = np.append(c3s, np.array(slices_test), axis = 0)
  
    segments = np.array(segments)
    c3s = np.array(c3s)
    return c3s, segments

# function to calculate mask density from mask area
def getDensity(image, mask, area, label=1):
  if image.shape != (len(image),1,...):
    mask = np.squeeze(mask)

  return float(np.mean(image[np.where(mask == 1)]))


# function to calculate mask area
def getArea(image, mask, area, label=1, thresholds = None):
  sMasks = (mask == label)
  threshold = np.logical_and(image > (thresholds[0]), image <  (thresholds[1]))
  tmask = np.logical_and(sMasks, threshold)
  return np.count_nonzero(tmask) * area

#classes
# custom dataset
  
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

#main
#Loading the data
# change this if not using colabs
path =   "/home/olivia/Documents/Internship_sarcopenia/176-215bone.npz" 
data = np.load(path, allow_pickle=True)

slices = data['slices']
bone = data['bone_masks']
ids = data['ids']
areas = data['pixel_areas']
#%%
#h
# preprocess slices and bone masks
slices_processed, masks_processed = preprocess(slices, bone)
# define test transforms
test_transform = A.Compose([A.Resize(260, 260), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensor()])

# create dataset object
test_dataset = H_custom_Dataset(slices_processed, masks_processed, transform = test_transform)
test_dataloader = DataLoader(test_dataset, batch_size = 8, num_workers = 2, shuffle = False)


#%%
#initilaise and load the model
from torchvision import models
model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
model_path = "/home/olivia/Documents/Internship_sarcopenia/model_state_dict_300_FL_testing.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#%%
# generating test predictions from model
c3s, test_predictions = test(model, test_dataloader)   
test_predictions = torch.from_numpy(test_predictions)
# removing the bone masks from the predictions
segment_pred_slb = np.logical_and(test_predictions, bone[:,np.newaxis,...])
segment_pred_slb = (segment_pred_slb.float()).numpy()
test_predictions = (test_predictions.float()).numpy()


#%%
#skeletal muscle area and density
ct_scans = np.array(slices)#or slices processed
pixel_areas = np.array(areas)*(0.1*0.1)#cm^2
## Note that the areas have some thresholds applied that are from the literature
extractionDict = {"sma" : partial(getArea, thresholds=(-30, +130)), 
				  "smd" : partial(getDensity)
				 }
feat_list_1 = ["sma","smd"]
# Extract features from slices_processed
feature_list_net = []
for i in range(0,len(ct_scans)):
  feature_list_net.append([extractionDict[a](ct_scans[i], segment_pred_slb[i], pixel_areas[i]) for a in feat_list_1])

sma = np.array(feature_list_net)[:,0]
mean_area = np.mean(sma)
area_sd = ndimage.standard_deviation(sma)
print(mean_area, "cm^2 ", "sd: ", area_sd)

smd = np.array(feature_list_net)[:,1]#sm attenuation
mean_density = np.mean(smd)
den_sd = ndimage.standard_deviation(smd)


#save the smas and smds
array = np.transpose(np.array(feature_list_net))
print(array)
df = pd.DataFrame(array, index= feat_list_1, columns=ids).T
df.to_excel(excel_writer = "/home/olivia/Documents/Internship_sarcopenia/muscle_area_and_density_all_slb_abstract.xlsx")

#display the predictions
fig=plt.figure(figsize=(10, 120))
ax = []
columns = 4
rows = int(len(slices)/columns + 0.5)
for i in range(0, len(slices)):
  ax.append(fig.add_subplot(rows,columns, i+1))
  segment_pred_slb[i][segment_pred_slb[i]==0] = np.nan
  plt.imshow(c3s[i,0,...], cmap="gray")
  plt.imshow(segment_pred_slb[i,0,...], cmap = "autumn", alpha = 0.5)
  #plt.imshow(test_lab[i,0,...], alpha= 0.5)
  ax[-1].set_title("Network test:" + ids[i])
  plt.axis("off")
plt.tight_layout(True)
plt.show()
fig.savefig("/home/olivia/Documents/Internship_sarcopenia/my_figure.png")

fig2 = plt.figure()
ax = []
bone = bone.astype(int)
print(np.unique(bone))
bone = np.where((bone==0)|(bone==1), bone^1, bone)
bone = bone.astype(float)
#print(bone)
ax.append(fig2.add_subplot(1,4,1))
plt.imshow(c3s[5,0,...], cmap="gray")
segment_pred_slb[5][segment_pred_slb[5]==0] = np.nan
test_predictions[5][test_predictions[5]==0] = np.nan
plt.axis('off')
ax.append(fig2.add_subplot(1,4,2))
plt.imshow(c3s[5,0,...], cmap="gray")
plt.imshow(segment_pred_slb[5,0,...], cmap = "autumn", alpha = 0.6)
plt.axis('off')
ax.append(fig2.add_subplot(1,4,3))
plt.imshow(c3s[5,0,...], cmap="gray")
plt.imshow(test_predictions[5,0,...], cmap = "autumn", alpha = 0.6)
plt.axis('off')
bone[5][bone[5] == 0] = np.nan
ax.append(fig2.add_subplot(1,4,4))
plt.imshow(c3s[5,0,...], cmap="gray")
plt.imshow(bone[5], cmap = "cool", alpha = 0.5)
plt.axis('off')
plt.savefig("/home/olivia/Documents/Internship_sarcopenia/my_figure2.png")

# %%
