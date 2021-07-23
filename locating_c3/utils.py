#utils
#created: 20/07/2021
#hermione

import numpy as np
import scipy.ndimage as nd

def GetSliceNumber(segment):
  slice_number = []
  max_range = len(segment)
  for x in range(0,max_range):
    seg_slice = segment[x,...]
    val = np.sum(seg_slice)
    if val != 0:
      slice_number.append(x)
  print(slice_number)
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