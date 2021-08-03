#utils
#created: 20/07/2021
#hermione

import numpy as np
import scipy.ndimage as nd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch


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
    max_range = len(target)
    for x in range(0,max_range):
        seg_slice_2 = target[x,:,:]
        val = np.sum(seg_slice_2)
        if val != 0:
            slice_number = x

    return coords

def Guassian(inp: np.ndarray):
  gauss = nd.gaussian_filter(inp)
  return gauss

def PrintSlice(input, targets):
    slice_no = GetSliceNumber(targets)
    print("slice no: ", slice_no)
    print("input shape: ", input.shape)
    plt.imshow(input[slice_no,...], cmap = "gray")
    #for i in range(len(targets)):
        #targets[i,...,0][targets[i,...,0] == 0] = np.nan
    plt.imshow(targets[slice_no,...], cmap = "cool", alpha = 0.5)
    plt.axis('off')
    plt.show()

def projections(inp: torch.tensor, msk: torch.tensor):
  cor, sag, ax = 0,1,2
  inp = inp.cpu().detach().numpy()
  msk = msk.cpu().detach().numpy()
  coronal = np.array(np.max(inp, axis = cor), np.average(inp, axis = cor), np.std(inp, axis=cor), np.max(msk, axis = cor))
  sagital = np.array(np.max(inp, axis = sag), np.average(inp, axis = sag), np.std(inp, axis=sag), np.max(msk, axis = sag))
  axial = np.array(np.max(inp, axis = ax), np.average(inp, axis = ax), np.std(inp, axis=ax), np.max(msk, axis = ax))
  return coronal, sagital, axial

