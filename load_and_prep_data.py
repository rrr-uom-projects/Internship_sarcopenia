#File to load in and store nifty images
#updated: 07/07/2021
#by hermione

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

def crop(slices_arr, masks_arr):
  slices_cropped = []
  masks_cropped = []
  for i in range(0,len(slices_arr)):
      crop_slice = slices_arr[i][0:250, 0:512]#(slices[i].shape[1]-100)
      #mask_slice = masks[i][0:250, 0:512]
      ret,threshold = cv2.threshold(crop_slice,200,250,cv2.THRESH_TOZERO)
      coords = ndimage.measurements.center_of_mass(threshold)
      size = 130
      x_min = int(((coords[0] - size)+126)/2)
      x_max = int(((coords[0] + size)+386)/2)
      y_min = int(((coords[1] - size)+126)/2)
      y_max = int(((coords[1] + size)+386)/2)
      #x_min = 126
      #x_max = 386
      #y_min = 126
      #y_max = 386
      crop_image = slices_arr[i][x_min:x_max, y_min:y_max]
      crop_seg = masks_arr[i][x_min:x_max, y_min:y_max]
      slices_cropped.append(crop_image)
      masks_cropped.append(crop_seg)
      return slices_cropped, masks_cropped

window = 350
level = 50

slices_array = []
mask_array = []
id_array = []
pixel_areas = []

for i in range(90,143):
  if i != 100 and i != 114 and i != 116 and i != 136 and i != 137: #patients 24 and 25 did not have rt scans
    path_ct = "/content/" + str(i) + ".nii"
    path_seg = "/content/" + str(i) + "_seg.nii"
    ct_scan = sitk.ReadImage(path_ct, imageIO="NiftiImageIO")
    segment = sitk.ReadImage(path_seg, imageIO="NiftiImageIO")
    slice_no = GetSliceNumber(segment)
    pixel_areas.append(((ct_scan.GetSpacing())[0])*((ct_scan.GetSpacing())[1]))
    ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_no,:,:]
    ct_slice = ct_slice.astype(float)
    slices_array.append(ct_slice)
    mask = sitk.GetArrayFromImage(segment)[slice_no,:,:]
    mask = mask.astype(float)
    #mask[mask_array == 0] = np.nan
    mask_array.append(mask)
    id_array_element = "01-00" + str(i)
    id_array.append(id_array_element)

print(pixel_areas)
slices_array = np.asarray(slices_array)
mask_array = np.asarray(mask_array)
id_array = np.asarray(id_array)
area_array = np.asarray(pixel_areas)

printTrainingDataForPatient(12)

print(slices_array.shape)

data = np.load("/content/training_data_h_c3_cropped.npz", allow_pickle=True)
slices = data['slices']
masks = data['masks']
ids = data['ids']
pixel_areas = data['pixel_areas']
print(slices.shape)

data_l = np.load("/content/first_50_test_o.npz", allow_pickle=True)
slices_l = data_l['slices']
masks_l = data_l['masks']
ids_l = data_l['ids']
pixel_areas_l = data_l['pixels']
#print(list(data_l.keys()))
#print(slices_l)
slices_l_new = np.array(np.delete(slices_l, [0]))
print(slices_l_new[0].shape)

slices_combined = np.concatenate((slices_l, slices))
masks_combined = np.concatenate(masks_l, masks)
ids_combined = np.concatenate(ids_l, ids)
pixel_areas_combined = np.concatenate(pixel_areas_l, pixel_areas)
print(slices_combined.shape)
 
#change names depending on whether one or two peoples data being uploaded
slices_cropped, masks_cropped = crop(slices_combined, masks_combined)

total_c3s = np.savez("/content/total_c3s", slices = slices_cropped, masks = masks_cropped, ids=id_array, pixel_areas = area_array)


for i in range(0, len(slices_cropped)):
  PrintTrainingDataLiv(slices_cropped, masks_cropped, id_array, i)

printTrainingData(36)