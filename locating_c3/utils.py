#utils
#created: 20/07/2021
#hermione

import numpy as np

def GetSliceNumber(segment):
  slice_number = 0
  max_range = len(segment)
  for x in range(0,max_range):
    seg_slice_2 = segment[x,:,:]
    val = np.sum(seg_slice_2)
    if val != 0:
      slice_number = x
  return slice_number

def GetTargetCoords(target):
    coords = []
    max_range = len(target)
    for x in range(0,max_range):
        seg_slice_2 = target[x,:,:]
        val = np.sum(seg_slice_2)
        if val != 0:
            slice_number = x

    return coords