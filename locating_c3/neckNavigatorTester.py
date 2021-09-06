# 28/07/2021
# Hermione Warr and Olivia Murray
# Test function for neckNavigator 

#imports 
import torch
import numpy as np
from utils import setup_model, flat_softmax
from neckNavigator import neckNavigator

def neckNavigatorTest2(model_dir, test_dataloader, device):
  model = setup_model(neckNavigator(), model_dir, device, load_prev = True, eval_mode=True)
  segments = []
  c3s = []
  GTs =[]
  for batch_idx, test_data in enumerate(test_dataloader):
    test_em, test_lab = test_data[0].to(device), test_data[1]
    test_em = test_em.type(torch.FloatTensor)
    test_em = test_em.to(device)
    output = model(test_em)
    output = flat_softmax(output)
    #print("output shape: ", output.shape)
    test_output = output.squeeze().cpu().detach().numpy()
    #print("test out: ",test_output.shape, np.max(test_output), np.min(test_output))
    #sigmoid = 1/(1 + np.exp(-test_output))
    segment = test_output.astype(np.float) #for heatmaps
    #segment = (segment > 0.5).astype(np.float)
    GTs.append(test_lab.squeeze().numpy())
    c3s.append(test_em.squeeze().cpu().detach().numpy())
    segments.append(segment)

  segments = np.asarray(segments)
  c3s = np.asarray(c3s)
  GTs = np.asarray(GTs)
  print("segments: ", segments.shape)
  print("c3s: ", c3s.shape)
  print("gts: ", GTs.shape)

  return c3s, segments, GTs