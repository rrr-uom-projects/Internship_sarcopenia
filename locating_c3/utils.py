#utils
#created: 20/07/2021
#hermione

import numpy as np
import scipy.ndimage as nd
from scipy.ndimage.measurements import center_of_mass
import torch
import matplotlib.pyplot as plt

def GetSliceNumber(segment):
  slice_number = []
  max_range = len(segment)
  for x in range(0,max_range):
    seg_slice = segment[x,...]
    val = np.sum(seg_slice)
    if val != 0:
      slice_number.append(x)
  return int(np.average(slice_number))

def GetTargetCoords(target):
    coords = []
    target = np.asarray(target)
    coords  = center_of_mass(target)
    #range = target.shape
    # for z in range(0,range[0]):
    #     seg_slice_2 = target[z,:,:]
    #     val = np.sum(seg_slice_2)
    #     if val != 0:
    #         coords.append(z)
    # for x in range(0,range[1]):
    #     seg_slice_2 = target[:,x,:]
    #     val = np.sum(seg_slice_2)
    #     if val != 0:
    #         coords.append(x)
    # for y in range(0,range[2]):
    #     seg_slice_2 = target[...,y]
    #     val = np.sum(seg_slice_2)
    #     if val != 0:
    #         coords.append(y)
    return coords

def Guassian(inp: np.ndarray):
  gauss = nd.gaussian_filter(inp)
  return gauss

def PrintSlice(input, targets):
    slice_no = GetSliceNumber(targets)
    print()
    plt.imshow(input[slice_no,...], cmap = "gray")
    #for i in range(len(targets)):
        #targets[i,...,0][targets[i,...,0] == 0] = np.nan
    plt.imshow(targets[slice_no,...], cmap = "cool", alpha = 0.5)
    plt.axis('off')
    plt.show()

def projections(inp, msk):
  cor, sag, ax = 1,2,0
  # if inp.type == torch.tensor:
  #   inp = inp.cpu().detach().numpy()
  #   msk = msk.cpu().detach().numpy()
  coronal = (np.max(inp, axis = cor), np.average(inp, axis = cor), np.std(inp, axis=cor))
  # holder = np.zeros((*(inp.shape), 3))
  # for i, img in enumerate(coronal):
  #       holder[..., i] = coronal
  # print(coronal.shape)
  sagital = (np.max(inp, axis = sag), np.average(inp, axis = sag), np.std(inp, axis=sag))
  axial = (np.max(inp, axis = ax), np.average(inp, axis = ax), np.std(inp, axis=ax))
  #coronal = np.max(inp, axis = cor)
  #sagital = np.max(inp, axis = sag)
  #axial = np.max(inp, axis = ax)
  cor_mask = np.max(msk, axis = cor)
  sag_mask = np.max(msk, axis = sag)
  ax_mask = np.max(msk, axis = ax)
  
  fig = plt.figure(figsize=(8, 8))
  ax = []
  columns = 3
  rows = 1
  images = (coronal,sagital,axial)
  masks = (cor_mask, sag_mask, ax_mask)
  for i in range(columns*rows):
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].set_title("ax:"+str(i))
    plt.imshow(images[i], cmap="gray")
    plt.imshow(masks[i], cmap="cool", alpha=0.5)
  plt.show()
  return coronal, sagital, axial

