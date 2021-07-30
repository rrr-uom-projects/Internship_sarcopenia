# 28/07/2021
# Hermione Warr and Olivia Murray
# Test function for neckNavigator 

#imports 
import torch
import numpy as np

device='cuda:0'
class neckNavigatorTest:
    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader
        
    def __getitem__(self, idx):
        self.model.eval()
        segments = []
        c3s = []
        for batch_idx, test_data in enumerate(self.test_dataloader):
             
           test_em, test_lab = test_data[0].to(device), test_data[1].to(device)
           test_em = test_em.type(torch.FloatTensor)
           output = self.model(test_em)
           print("output shape: ", output.shape)
           test_output = output.squeeze().cpu().detach().numpy()
           print(test_em.shape)
           sigmoid = 1/(1 + np.exp(-test_output))
           segment = (sigmoid > 0.5)
           print("np unique segment: ", np.unique(segment))
        #    if int == 0:
        #       segments == segment
        #       c3s == test_em.cpu().detach().numpy()
        #    else:
        #        print("segments size :", np.array(segments).shape, "segment size: ", np.array(segment).shape)
        #        segments = np.append(segments, np.array(segment))
        #        c3s = np.append(c3s, np.array(test_em.cpu().detach().numpy())
           c3s.append(test_em.squeeze().cpu().detach().numpy())
           segments.append(segment)

        segments = np.asarray(segments)
        c3s = np.asarray(c3s)
        return c3s, segments





         


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