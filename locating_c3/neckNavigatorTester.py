# 28/07/2021
# Hermione Warr and Olivia Murray
# Test function for neckNavigator 

#imports 
import torch
import numpy as np

def neckNavigatorTest1(model, test_dataloader, device):
  model.eval()
  segments = []
  c3s = []
  GTs =[]
  for batch_idx, test_data in enumerate(test_dataloader):
    test_em, test_lab = test_data[0].to(device), test_data[1]
    test_em = test_em.type(torch.FloatTensor)
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





         


    """
    model.eval()
    segments = []
    c3s = []

    for int, data in enumerate(test_dataloader):
        slices_test = data[0].to(device)
        slices_test = slices_test.type(torch.float32)
        output = model(slices_test)["out"]
        print("output shape: ", output.shape)
        test_ouput = output.detach().cpu()
        slices_test = slices_test.detach().cpu()
        sigmoid = 1/(1 + np.exp(-test_ouput))
        segment = (sigmoid > 0.5).float()
        print(np.unique(segment))
        #print(int)
        if int == 0:
          segments = segment
          c3s = slices_test
        else:
          segments = np.append(segments, np.array(segment), axis = 0)
          c3s = np.append(c3s, np.array(slices_test), axis = 0)
  
    segments = np.array(segments)
    c3s = np.array(c3s)
    return c3s, segments
    """