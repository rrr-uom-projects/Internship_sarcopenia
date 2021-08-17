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

def Guassian(inp: np.ndarray):
  gauss = nd.gaussian_filter(inp,3)
  return gauss

def setup_model(model, checkpoint_dir, device, load_prev = False, eval_mode = False):
  model.to(device)
  if load_prev == True:
    model.load_best(checkpoint_dir, logger=None)
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

  def arrange(input, ax):
    #to return the projection in whatever order.
    av = np.average(input, axis = ax)
    mx = np.max(input, axis = ax)
    std = np.std(input, axis=ax)
    ord_list = [mx,av,std]
    ord_list[:] = [ord_list[i] for i in proj_order]
    out = np.stack((ord_list), axis=2)
    return out
  #print(inp.shape)
  axial = arrange(inp, axi)
  coronal = arrange(inp, cor)
  sagital = arrange(inp, sag)

  ax_mask = np.max(msk, axis = axi)
  cor_mask = np.max(msk, axis = cor)
  sag_mask = np.max(msk, axis = sag)
  #flip right way up
  coronal = coronal[::-1]
  sagital = sagital[::-1]
  cor_mask = cor_mask[::-1]
  sag_mask = sag_mask[::-1]
  images = (axial,coronal,sagital)
  masks = (ax_mask,cor_mask, sag_mask)
  #print(coronal.shape)
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
    gts = gts.cpu().detach().numpy()[0]
    masks = masks.cpu().detach().numpy()[0]
  distances = []
  for i in range(len(gts)):
    #gts[i][gts[i] == np.nan] = 0
    #masks[i][masks[i] == np.nan] = 0
    #print(np.unique(masks[i]))
    #print(np.unique(gts[i]))
    gt_coords = GetTargetCoords(gts[i])
    msk_coords = GetTargetCoords(masks[i])
    print(gt_coords)
    print(msk_coords)
    distances.append(np.abs(gt_coords[0]-msk_coords[0]))
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