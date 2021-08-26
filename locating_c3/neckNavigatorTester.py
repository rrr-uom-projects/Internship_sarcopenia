# 28/07/2021
# Hermione Warr and Olivia Murray
# Test function for neckNavigator 

#imports 
import torch
import numpy as np
from utils import setup_model
from neckNavigator import neckNavigator

def neckNavigatorTest2(model_dir, test_dataloader, device):
  model = setup_model(neckNavigator(filter_factor=2, targets = 1, in_channels = 1), model_dir, device, load_prev = True,eval_mode=True)
  #test_dataloader = torch.load(test_dataloader_dir)
  #model.eval()
  segments = []
  c3s = []
  GTs =[]
  for batch_idx, test_data in enumerate(test_dataloader):
    test_em, test_lab = test_data[0].to(device), test_data[1]
    test_em = test_em.type(torch.FloatTensor)
    test_em = test_em.to(device)
    output = model(test_em)
    #print("output shape: ", output.shape)
    test_output = output.squeeze().cpu().detach().numpy()
    #print("test out: ",test_output.shape, np.max(test_output), np.min(test_output))
    sigmoid = 1/(1 + np.exp(-test_output))
    segment = sigmoid.astype(np.float) #for heatmaps
    #segment = (sigmoid > 0.5).astype(np.float)
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

def neckNavigatorTest1(model, model_dir, test_dataloader, device):
  model = setup_model(model, model_dir, device, load_prev=True, eval_mode=True)
  segments = []
  c3s = []
  GTs =[]
  for batch_idx, test_data in enumerate(test_dataloader):
    test_em, test_lab = test_data[0].type(torch.FloatTensor).to(device), test_data[1]
    #test_em = test_em.type(torch.FloatTensor)
    output = model(test_em)
    #print("output shape: ", output.shape)
    test_output = output.squeeze().cpu().detach().numpy()
    #print("test out: ",test_output.shape, np.max(test_output), np.min(test_output))
    sigmoid = 1/(1 + np.exp(-test_output))
    segment = sigmoid.astype(np.float) #for heatmaps
    #segment = (sigmoid > 0.5).astype(np.float)
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

class neckNavigatorTest:
    def __init__(self, model, test_dataloader, device):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        
    def test(self, idx):
        self.model.eval()
        segments = []
        c3s = []
        GTs =[]
        for batch_idx, test_data in enumerate(self.test_dataloader):
           test_em, test_lab = test_data[0].to(self.device), test_data[1]
           test_em = test_em.type(torch.FloatTensor)
           output = self.model(test_em)
           #print("output shape: ", output.shape)
           test_output = output.squeeze().cpu().detach().numpy()
           #print("test out: ",test_output.shape, np.max(test_output), np.min(test_output))
           sigmoid = 1/(1 + np.exp(-test_output))
           segment = sigmoid.astype(np.float) #for heatmaps
           #segment = (sigmoid > 0.5).astype(np.float)
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
