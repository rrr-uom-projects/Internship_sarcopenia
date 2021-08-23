#utils
#created: 20/07/2021
#hermione

from SimpleITK.SimpleITK import Modulus
import numpy as np
import scipy.ndimage as nd
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import torch
import torchvision.transforms as T
import torchvision.io.image as tim


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

def slice_preds(masks):
  slice_nos = []
  for i in range(len(masks)):
    slice_nos.append(GetTargetCoords(masks[i])[2])
  return np.asarray(slice_nos)

def Guassian(inp: np.ndarray):
  gauss = nd.gaussian_filter(inp,3)
  return gauss

def setup_model(model, checkpoint_dir, device, load_prev = False, load_best = False, eval_mode = False):
  model.to(device)
  if load_prev == True:
    model.load_previous(checkpoint_dir, logger=None)
  if load_best == True:
    model.load_best(checkpoint_dir, logger = None)
  for param in model.parameters():
      param.requires_grad = True
  if eval_mode:
    model.eval()
  return model

def PrintSlice(input, targets, show = False):
    slice_no = GetSliceNumber(targets)
    print("slice no: ", slice_no)
    #print("input shape: ", input.shape)
    plt.imshow(input[slice_no,...], cmap = "gray")
    #for i in range(len(targets)):
        #targets[i,...,0][targets[i,...,0] == 0] = np.nan
    plt.imshow(targets[slice_no,...], cmap = "cool", alpha = 0.5)
    plt.axis('off')
    if show:
      plt.show()
      plt.savefig("Slice.png")

def arrange(input, ax, proj_order):
    #to return the projection in whatever order.
    av = np.average(input, axis = ax)
    mx = np.max(input, axis = ax)
    std = np.std(input, axis=ax)
    ord_list = [mx,av,std]
    ord_list[:] = [ord_list[i] for i in proj_order]
    out = np.stack((ord_list), axis=2)
    return out

def base_projections(inp, msk):
  axi,cor,sag = 0,1,2
  proj_order = [2,1,0]
  axial = arrange(inp, axi, proj_order)
  coronal = arrange(inp, cor, proj_order)
  sagital = arrange(inp, sag, proj_order)
  ax_mask = np.max(msk, axis = axi)
  cor_mask = np.max(msk, axis = cor)
  sag_mask = np.max(msk, axis = sag)
  #flip right way up
  coronal = coronal[::-1]
  sagital = sagital[::-1]
  cor_mask = cor_mask[::-1]
  sag_mask = sag_mask[::-1]
  image = (axial,coronal,sagital)
  mask = (ax_mask,cor_mask, sag_mask)
  return image, mask

def projections(inp, msk, order, type = "numpy", show = False, save_name = None):
  axi,cor,sag = 0,1,2
  proj_order = order
  if type == "tensor":
     inp = inp.cpu().detach().squeeze().numpy()
     msk = msk.cpu().detach().squeeze().numpy().astype(float)
     if len(inp.shape) == 4:
       inp = inp[0]
       msk = msk[0]
     #print(msk.shape)
  images, masks = base_projections(inp, msk)
  fig = plt.figure(figsize=(48, 16))
  ax = []
  columns = 3
  rows = 1
  for i in range(columns*rows):
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title("ax:"+str(i))
    plt.imshow(images[i])
    #for j in range(len(masks[i])):
        #masks[i][j][masks[i][j] == 0] = np.nan
    plt.imshow(masks[i], cmap="cool", alpha=0.5)
    plt.axis('off')
  if save_name is not None:
    plt.savefig("projections" + str(save_name) + ".png")
  if show: 
    plt.show()
  return fig

#classs inbalence
#ratio of no of 1s over no of 0s. averaged
def classRatio(masks):
    ratio = []
    for i in range(len(masks)):
        no_of_0s = (masks[i] == 0).sum()
        no_of_1s = (masks[i] == 1).sum()
        ratio.append(no_of_1s/no_of_0s)
    average_ratio = np.mean(ratio)
    print(average_ratio)
    weights = (1/average_ratio) #penalise the network more harshly for getting it wrong
    print(weights)

def euclid_dis(gts, masks, is_tensor = False):
  #quantifies how far of the network's predictions are
  if is_tensor:
    gt = gts
    msk = masks
    # pred_vox = torch.tensor([np.unravel_index(torch.argmax(gts[i, 0]), gts.size()[2:]) for i in range(gts.size(0))]).type(torch.FloatTensor)
    # gt_vox = torch.tensor([np.unravel_index(torch.argmax(masks[i, 0]), masks.size()[2:]) for i in range(masks.size(0))]).type(torch.FloatTensor)
    # print(pred_vox, gt_vox)
    # print("vox_coord dff: ",torch.abs(gt_vox[2]-pred_vox[2]))
    gts = gt.cpu().detach().numpy()[0]
    masks = msk.cpu().detach().numpy()[0]
  distances = []
  for i in range(len(gts)):
    print(np.max(masks[i]))
    print(np.max(gts[i]))
    gt_coords = GetTargetCoords(gts[i])
    msk_coords = GetTargetCoords(masks[i])
    print(gt_coords)
    print(msk_coords)
    distances.append(np.abs(gt_coords[2]-msk_coords[2]))
  distances = np.array(distances)
  print(np.average(distances))
  return distances

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue())
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def get_data(path): 
    data = np.load(path)
    inputs = data['inputs']
    targets = data['masks']
    ids = data['ids']
    return np.asarray(inputs), np.asarray(targets), np.asarray(ids)


def display_input_data(path, type = 'numpy' ,save_name = 'gauss_data', show = False):
  inps,msks,ids = get_data(path)
  data_size = len(inps)
  axi,cor,sag = 0,1,2
  proj_order = [2,1,0]
  images = []
  targets = []
  if type == "tensor":
     inp = inps.cpu().detach().squeeze().numpy()
     msk = msks.cpu().detach().squeeze().numpy().astype(float)
       
  for j in range(data_size):
    inp = inps[j]
    msk = msks[j]
    image, mask = base_projections(inp, msk)
    images.append(image)
    targets.append(mask)

  fig = plt.figure(figsize=(200, 400))
  ax = []
  columns = 9
  rows = 1 + data_size/3
  
  for l in range(1, data_size +1):
    image = images[l-1]
    target =  targets[l-1]
    for i in range(1,4):
      # create subplot and append to ax
      print(l,i)
      ax.append(fig.add_subplot(rows, columns, l*i))
      ax[-1].set_title(str(l) + ',' + str(i-1))
      plt.imshow(image[i-1])
      plt.imshow(target[i-1], cmap="cool", alpha=0.5)
      plt.axis('off')
      
  path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/pic_'
  plt.savefig(path + str(save_name) + '.png')
  return fig

#data_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_gauss2.npz'
#display_input_data(data_path)

def display_net_test(inps, msks, gts):
  images, targets, preds = [],[],[]
  data_size = len(inps)
  slice_no_preds = slice_preds(msks)
  print("test data size: ", data_size)
  for i in range(data_size):
    image, gt = base_projections(inps[i], gts[i])
    images.append(image)
    targets.append(gt)
    _, pred = base_projections(inps[i], msks[i])
    preds.append(pred)
  #make the figure
  fig = plt.figure(figsize=(200, 400))
  ax = []
  columns = 3
  rows = 2*data_size
  for l in range(1, data_size +1):
    image = images[l-1]
    target =  targets[l-1]
    pred = preds[l-1]
    for i in range(1,4):
      #create gt subplot 
      print((2*l),i)
      ax.append(fig.add_subplot(rows, columns, (2*l-1)*i))
      ax[-1].set_title("GT " + str(l) + ',' + str(i-1))
      plt.imshow(image[i-1])
      plt.imshow(target[i-1], cmap="cool", alpha=0.5)
    for i in range(1,4):
      #mask subplot
      print((2*l+1),i)
      ax.append(fig.add_subplot(rows, columns, (2*l)*i))
      ax[-1].set_title("pred " + str(l) + ',' + str(i-1))
      plt.imshow(image[i-1])
      plt.imshow(pred[i-1], cmap="cool", alpha=0.5)
      ax[-1].axhline(slice_no_preds, linewidth=2, c='y')
      ax[-1].text(0, slice_no_preds-5, "C3", color='w')
    plt.axis('off')
  path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/test_pic'
  plt.savefig(path + '.png')
  return fig