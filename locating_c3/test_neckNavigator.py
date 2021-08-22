#created by hermione on 16/08/2021
#to test the model at various stages

from utils import PrintSlice, get_data, projections, euclid_dis, display_net_test
from neckNavigatorTester import neckNavigatorTest1
from neckNavigator import neckNavigator
from neckNavigatorData import neckNavigatorDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from neckNavigatorUtils import k_fold_split_train_val_test

def main():

    # get data
    #data_path =  '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed_sphere.npz'
    data_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_gauss2.npz'
    data = get_data(data_path)
    inputs = data[0]
    targets = data[1]
    ids = data[2]

    val_workers = int(4)

    train_inds, val_inds, test_inds = k_fold_split_train_val_test(len(inputs), fold_num= 2)
    test_dataset = neckNavigatorDataset(inputs = inputs, targets = targets, image_inds = test_inds)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    
    #model_dir = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1/"
    #testdataloader_dir = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1/test_dataloader.pt"
    device = 'cuda:0'

    #tester = neckNavigatorTest2(model_dir, test_dataloader, device)
    checkpoint_dir = checkpoint_dir = "/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs"
    model = neckNavigator()
    tester = neckNavigatorTest1(model, checkpoint_dir, test_dataloader, device)

    #test_results = tester
    C3s, segments, GTs = tester
    
    print("gt info: ", len(GTs))
    print(GTs.shape)
    print("segs info: ", segments.shape)

    difference = euclid_dis(GTs, segments)
    print(difference)
    projections(C3s[1],segments[1], order = [1,2,0], show=True)
    projections(C3s[1],GTs[1], order = [1,2,0], show=True)
    #display_net_test(C3s, segments, GTs)

    return

if __name__ == '__main__':
    main()