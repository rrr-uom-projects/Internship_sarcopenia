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
    coords = center_of_mass(target)
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

def projections(inp, msk, order,  type = "numpy"):
  cor, sag, axi = 0,1,2
  proj_order = order
  if type == "tensor":
     inp = inp.cpu().detach().numpy()
     msk = msk.cpu().detach().numpy()

  def arrange(input, ax):
    #to return the projection in whatever order.
    av = np.average(input, axis = ax)
    mx = np.max(input, axis = ax)
    std = np.std(input, axis=ax)
    ord_list = [mx,av,std]
    ord_list[:] = [ord_list[i] for i in proj_order]
    out = np.stack((ord_list), axis=2)
    return out

  coronal = arrange(inp, cor)
  sagital = arrange(inp, sag)
  axial = arrange(inp, axi)
  #coronal = np.stack((np.average(inp, axis = cor), np.max(inp, axis = cor), np.std(inp, axis=cor)),axis=2)
  # holder = np.zeros((*(inp.shape), 3))
  # for i, img in enumerate(coronal):
  #       holder[..., i] = coronal
  # print(coronal.shape)
  #sagital = np.stack((np.average(inp, axis = sag), np.max(inp, axis = sag), np.std(inp, axis=sag)),axis=2)
  #axial = np.stack((np.average(inp, axis = ax), np.max(inp, axis = ax), np.std(inp, axis=ax)),axis=2)
  cor_mask = np.max(msk, axis = cor)
  sag_mask = np.max(msk, axis = sag)
  ax_mask = np.max(msk, axis = axi)
  
  fig = plt.figure(figsize=(8, 8))
  ax = []
  columns = 3
  rows = 1
  images = (coronal,sagital,axial)
  masks = (cor_mask, sag_mask, ax_mask)
  print(coronal.shape)

  for i in range(columns*rows):
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title("ax:"+str(i))
    plt.imshow(images[i])
    for j in range(len(masks[i])):
        masks[i][j][masks[i][j] == 0] = np.nan
    plt.imshow(masks[i], cmap="autumn", alpha=0.5)
    plt.axis('off')
  plt.show()
  return coronal, sagital, axial

