#File to load in and store nifty images
#updated: 07/07/2021
#by hermione
#%%
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage
import cv2
from decimal import Decimal

###*** load in niftis ***###
def do_it_urself_round(number, decimals = 0, is_array = False):
  float_num = number
  dec_num =  Decimal(str(number))
  Ddecimal = Decimal(str(decimals))
  round_num = round(dec_num, decimals)
  diff = np.abs(dec_num - round_num)
  round_diff = Decimal(str(0.5/(10**decimals)))
  if (diff == round_diff and dec_num >= round_num):
    round_num += (1/(10**(Ddecimal))) 
  if decimals == 0:
    return int(round_num)
  else:
    return float(round_num)

def GetSliceNumber(segment):
  slice_number = 0
  max_range = len(sitk.GetArrayFromImage(segment))
  for x in range(0,max_range):
    seg_slice_2 = sitk.GetArrayFromImage(segment)[x,:,:]
    val = np.sum(seg_slice_2)
    if val != 0:
      slice_number = x
  return slice_number

def crop2(slices_arr, masks_arr, bone_arr, sans_le_bone_arr):
  #taking a mean of the centre of image and centre of intensity
  slices_cropped = []
  masks_cropped = []
  bone_cropped = []
  mask_sans_le_bone_cropped = []
  for i in range(len(slices_arr)):
      crop_slice = slices_arr[i][0:250, 0:512]
      ret,threshold = cv2.threshold(crop_slice,200,250, cv2.THRESH_TOZERO)
      coords = ndimage.measurements.center_of_mass(threshold)
      size = 110 #try 110?
      x_min = int(((coords[0] - size)+126)/2)
      x_max = int(((coords[0] + size)+386)/2)
      y_min = int(((coords[1] - size)+126)/2)
      y_max = int(((coords[1] + size)+386)/2)
      crop_image = slices_arr[i][x_min:x_max, y_min:y_max]
      crop_seg = masks_arr[i][x_min:x_max, y_min:y_max]
      crop_bone = bone_arr[i][x_min:x_max, y_min:y_max]
      crop_sans = sans_le_bone_arr[i][x_min:x_max, y_min:y_max]
      slices_cropped.append(crop_image)
      masks_cropped.append(crop_seg)
      bone_cropped.append(crop_bone)
      mask_sans_le_bone_cropped.append(crop_sans)
  return np.asarray(slices_cropped), np.asarray(masks_cropped), np.asarray(bone_cropped), np.asarray(mask_sans_le_bone_cropped)

def extract_bone_masks(dcm_array, slice_number, threshold=(200+1024), radius=2, worldmatch=False):
    """
    Calculate 3D bone mask and remove from prediction. 
    Used to account for partial volume effect.
​
    Args:
        dcm_array - 3D volume
        slice number - slice where segmentation being performed
        threshold (in HU) - threshold above which values are set to 1 
        radius (in mm) - Kernel radius for binary dilation
    """
    img = dcm_array
    # Worldmatch tax
    # if worldmatch:
    #     img -= 1024
    # Apply threshold
    bin_filt = sitk.BinaryThresholdImageFilter()
    bin_filt.SetOutsideValue(1)
    bin_filt.SetInsideValue(0)
    bin_filt.SetLowerThreshold(-1024)
    bin_filt.SetUpperThreshold(threshold)
    bone_mask = bin_filt.Execute(img)
    pix = bone_mask.GetSpacing()
    # Convert to pixels
    pix_rad = [int(radius//elem) for elem in pix]
    # Dilate mask
    dil = sitk.BinaryDilateImageFilter()
    dil.SetKernelType(sitk.sitkBall)
    dil.SetKernelRadius(pix_rad)
    dil.SetForegroundValue(1)
    dilated_mask = dil.Execute(bone_mask)
    arr = sitk.GetArrayFromImage(dilated_mask)[slice_number]
    return np.logical_not(arr)

def load_n_save(dir, save_path, is_worldmatch = True):
  names = [y.split('_')[0] for y in os.listdir(dir)]
  patient_paths = [os.path.join(dir, x) for x in os.listdir(dir)]
  # Load input and target
  slices = []
  masks = []
  pixel_areas = []
  ids = []
  bone_masks = []
  masks_slb = []
  slice_nos = []

  for i in range(10):
    path_seg = dir + f'/{i+1}_aseg.nii.gz'
    path_ct = dir + f'/{i+1}_ct.nii'
    segment = sitk.ReadImage(path_seg, imageIO="NiftiImageIO")
    slice_no = GetSliceNumber(segment)
    slice_nos.append(slice_no)
    mask = sitk.GetArrayFromImage(segment)[slice_no,:,:]
    masks.append(mask.astype(float))#.astype(float)
    ct_scan = sitk.ReadImage(path_ct, imageIO="NiftiImageIO")
    pixel_areas.append(((ct_scan.GetSpacing())[0])*((ct_scan.GetSpacing())[1]))
    ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_no]
    boneMask = extract_bone_masks(ct_scan, slice_no)
    bone_masks.append(boneMask)
    if is_worldmatch: ct_slice -=1024
    print(np.max(ct_slice), np.min(ct_slice))
    slices.append(ct_slice.astype(float))
    masks_slb.append(np.array(np.logical_and(masks, boneMask))) #not right for some reason
    
   
  slices = np.array(slices)
  masks = np.array(masks)
  bone_masks = np.array(bone_masks)
  masks_slb = np.squeeze(np.array([masks_slb]), axis=0)
  slice_nos = np.array(slice_nos)
  print(len(masks), len(bone_masks))
  print(masks_slb.shape)
  cropped_slices, cropped_masks, cropped_bone, cropped_masks_slb = crop2(slices, masks, bone_masks, masks_slb)

  #data = {'slice': np.array(slices), 'masks': np.array(masks),'masks_slb': np.array(masks_slb), 'bone_masks': np.array(bone_masks), 'areas': np.array(pixel_areas), 'ids':ids}  
  np.savez(save_path, slices = cropped_slices, masks = cropped_masks, bone = cropped_bone, masks_slb = cropped_masks_slb, ids = ids, areas = pixel_areas )
  return #data

path = '/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/extra_segs'
save_path = '/home/hermione/Documents/Internship_sarcopenia/sarcopenia_model/save_extras.npz'
load_n_save(path, save_path)
data = np.load(save_path, allow_pickle=True)
print([*data.keys()])
slices = data['slices']
masks_slb = data['masks_slb']
bone = data['bone']
masks = data['masks']
print(slices.shape)
print(np.unique(masks_slb))
for i in range(len(slices)):
  plt.imshow(slices[i], cmap = plt.cm.gray)
  masks[i][masks[i]==0.0]=np.nan
  plt.imshow(masks[i], cmap = plt.cm.autumn, alpha= 0.5)
  bone[i] = bone[i].astype(float)
  bone[i][bone[i]==1.0] = np.nan
  plt.imshow(bone[i], cmap = plt.cm.cool, alpha=0.5)
  plt.axis('off')
  plt.show()
  plt.savefig(f'extra_im{i}.png')
#%%
# def sitk_show(img, title=None, margin=0.05, dpi=40 ):
#     nda = sitk.GetArrayFromImage(img)
#     spacing = img.GetSpacing()
#     figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
#     extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#     ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

#     plt.set_cmap("gray")
#     ax.imshow(nda,extent=extent,interpolation=None)
    
#     if title:
#         plt.title(title)
    
#     plt.show()


# def GetSliceNumber(segment):
#   slice_number = 0
#   max_range = len(sitk.GetArrayFromImage(segment))
#   for x in range(0,max_range):
#     seg_slice_2 = sitk.GetArrayFromImage(segment)[x,:,:]
#     val = np.sum(seg_slice_2)
#     if val != 0:
#       slice_number = x
#   return slice_number

# def printTrainingDataForPatient(patient_number):
#   if patient_number == 24 or patient_number == 25 or patient_number == 37: 
#     print("not valid patient number")
#   elif patient_number in range(90,143):
#     #if patient_number > 25: 
#      # patient_number = patient_number - 2
#     print("HNSCC Patient " + str(id_array[patient_number -90]))
#     plt.imshow(slices_array[patient_number-90], cmap=plt.cm.gray, vmin=(level/2 - window), vmax=(level/2 + window))
#     mask_array[patient_number-90][mask_array[patient_number-90] == 0] = np.nan
#     plt.imshow(mask_array[patient_number-90], cmap=plt.cm.autumn, alpha = 0.4)
#     plt.show()
#   else:
#     print("not valid patient number")

# def printTrainingData(patient_number):
#   if patient_number == 24 or patient_number == 25 or patient_number == 37:
#     print("not valid patient number")
#   elif patient_number in range(90,143):
#     slices = data['slices']
#     masks = data['masks']
#     ids = data['ids']
#     if patient_number > 24: 
#       patient_number = patient_number - 2
#     print("HNSCC Patient " + str(ids[patient_number-90]))
#     plt.imshow(slices[patient_number-1], cmap=plt.cm.gray, vmin=(level/2 - window), vmax=(level/2 + window))
#     masks[patient_number-1][masks[patient_number-1] == 0] = np.nan
#     plt.imshow(masks[patient_number-1], cmap=plt.cm.autumn, alpha = 0.4)
#     plt.show()
#   else:
#     print("not valid patient number")

# def PrintTrainingDataLiv(slicef_array, maskf_array, idf_array, i ):
#   print("HNSCC Patient " + str(idf_array[i]), "index: ", i)
#   plt.imshow(slicef_array[i], cmap = plt.cm.gray, vmin = level/2 - window, vmax = level/2 + window)
#   seg_slice = maskf_array[i]
#   seg_slice = seg_slice.astype(float)
#   seg_slice[seg_slice == 0] = np.nan
#   plt.imshow(seg_slice, cmap = plt.cm.autumn, alpha = 0.6)
#   plt.show()

# def crop(slices_arr, masks_arr):
#   slices_cropped = []
#   masks_cropped = []
#   for i in range(0,len(slices_arr)):
#       crop_slice = slices_arr[i][0:250, 0:512]#(slices[i].shape[1]-100)
#       #mask_slice = masks[i][0:250, 0:512]
#       ret,threshold = cv2.threshold(crop_slice,200,250,cv2.THRESH_TOZERO)
#       coords = ndimage.measurements.center_of_mass(threshold)
#       size = 130
#       x_min = int(((coords[0] - size)+126)/2) #xcrop = (size +386 -(126-size))/2
#       x_max = int(((coords[0] + size)+386)/2)
#       y_min = int(((coords[1] - size)+126)/2)
#       y_max = int(((coords[1] + size)+386)/2)
#       #x_min = 126
#       #x_max = 386
#       #y_min = 126
#       #y_max = 386
#       crop_image = slices_arr[i][x_min:x_max, y_min:y_max]
#       crop_seg = masks_arr[i][x_min:x_max, y_min:y_max]
#       slices_cropped.append(crop_image)
#       masks_cropped.append(crop_seg)
#       return slices_cropped, masks_cropped

# window = 350
# level = 50

# slices_array = []
# mask_array = []
# id_array = []
# pixel_areas = []

# for i in range(90,143):
#   if i != 100 and i != 114 and i != 116 and i != 136 and i != 137: #patients 24 and 25 did not have rt scans
#     path_ct = "/content/" + str(i) + ".nii"
#     path_seg = "/content/" + str(i) + "_seg.nii"
#     ct_scan = sitk.ReadImage(path_ct, imageIO="NiftiImageIO")
#     segment = sitk.ReadImage(path_seg, imageIO="NiftiImageIO")
#     slice_no = GetSliceNumber(segment)
#     pixel_areas.append(((ct_scan.GetSpacing())[0])*((ct_scan.GetSpacing())[1]))
#     ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_no,:,:]
#     ct_slice = ct_slice.astype(float)
#     slices_array.append(ct_slice)
#     mask = sitk.GetArrayFromImage(segment)[slice_no,:,:]
#     mask = mask.astype(float)
#     #mask[mask_array == 0] = np.nan
#     mask_array.append(mask)
#     id_array_element = "01-00" + str(i)
#     id_array.append(id_array_element)

# print(pixel_areas)
# slices_array = np.asarray(slices_array)
# mask_array = np.asarray(mask_array)
# id_array = np.asarray(id_array)
# area_array = np.asarray(pixel_areas)

# printTrainingDataForPatient(12)

# print(slices_array.shape)

# data = np.load("/content/training_data_h_c3_cropped.npz", allow_pickle=True)
# slices = data['slices']
# masks = data['masks']
# ids = data['ids']
# pixel_areas = data['pixel_areas']
# print(slices.shape)

# data_l = np.load("/content/first_50_test_o.npz", allow_pickle=True)
# slices_l = data_l['slices']
# masks_l = data_l['masks']
# ids_l = data_l['ids']
# pixel_areas_l = data_l['pixels']
# #print(list(data_l.keys()))
# #print(slices_l)
# slices_l_new = np.array(np.delete(slices_l, [0]))
# print(slices_l_new[0].shape)

# slices_combined = np.concatenate((slices_l, slices))
# masks_combined = np.concatenate(masks_l, masks)
# ids_combined = np.concatenate(ids_l, ids)
# pixel_areas_combined = np.concatenate(pixel_areas_l, pixel_areas)
# print(slices_combined.shape)
 
# #change names depending on whether one or two peoples data being uploaded
# slices_cropped, masks_cropped = crop(slices_combined, masks_combined)

# total_c3s = np.savez("/content/total_c3s", slices = slices_cropped, masks = masks_cropped, ids=id_array, pixel_areas = area_array)

# for i in range(0, len(slices_cropped)):
#   PrintTrainingDataLiv(slices_cropped, masks_cropped, id_array, i)

# printTrainingData(36)
