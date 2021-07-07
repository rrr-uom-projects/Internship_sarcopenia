import matplotlib.pyplot as plt
#from skimage.io import imread
#from __future__ import print_function
%matplotlib inline
import SimpleITK as sitk
#from skimage.io import imread
import numpy as np
import os
from scipy import ndimage
import cv2

def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()


def GetSliceNumber(segment):
  slice_number = 0
  max_range = len(sitk.GetArrayFromImage(segment))
  for x in range(0,max_range):
    seg_slice_2 = sitk.GetArrayFromImage(segment)[x,:,:]
    val = np.sum(seg_slice_2)
    if val != 0:
      slice_number = x
  return slice_number

def printTrainingDataForPatient(patient_number):
  if patient_number == 24 or patient_number == 25 or patient_number == 37: 
    print("not valid patient number")
  elif patient_number in range(90,143):
    #if patient_number > 25: 
     # patient_number = patient_number - 2
    print("HNSCC Patient " + str(id_array[patient_number -90]))
    plt.imshow(slices_array[patient_number-90], cmap=plt.cm.gray, vmin=(level/2 - window), vmax=(level/2 + window))
    mask_array[patient_number-90][mask_array[patient_number-90] == 0] = np.nan
    plt.imshow(mask_array[patient_number-90], cmap=plt.cm.autumn, alpha = 0.4)
    plt.show()
  else:
    print("not valid patient number")

def printTrainingData(patient_number):
  if patient_number == 24 or patient_number == 25 or patient_number == 37:
    print("not valid patient number")
  elif patient_number in range(90,143):
    slices = data['slices']
    masks = data['masks']
    ids = data['ids']
    if patient_number > 24: 
      patient_number = patient_number - 2
    print("HNSCC Patient " + str(ids[patient_number-90]))
    plt.imshow(slices[patient_number-1], cmap=plt.cm.gray, vmin=(level/2 - window), vmax=(level/2 + window))
    masks[patient_number-1][masks[patient_number-1] == 0] = np.nan
    plt.imshow(masks[patient_number-1], cmap=plt.cm.autumn, alpha = 0.4)
    plt.show()
  else:
    print("not valid patient number")

def PrintTrainingDataLiv(slicef_array, maskf_array, idf_array, i ):
  print("HNSCC Patient " + str(idf_array[i]), "index: ", i)
  plt.imshow(slicef_array[i], cmap = plt.cm.gray, vmin = level/2 - window, vmax = level/2 + window)
  seg_slice = maskf_array[i]
  seg_slice = seg_slice.astype(float)
  seg_slice[seg_slice == 0] = np.nan
  plt.imshow(seg_slice, cmap = plt.cm.autumn, alpha = 0.6)
  plt.show()

