#created by hermione on 16/08/2021
#to test the model at various stages


from numpy.lib.function_base import average
from utils import PrintSlice, projections, setup_model
from neckNavigatorTester import neckNavigatorTest2
from neckNavigator import neckNavigator
from neckNavigatorData import neckNavigatorDataset
from utils import get_data
from torch.utils.data import DataLoader
import numpy as np
import torch
from utils import euclid_dis
from neckNavigatorUtils import k_fold_split_train_val_test
def main():
    # get data
    
    data_path = data_path = '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed_sphere.npz'
    data = get_data(data_path)
    inputs = data[0]
    targets = data[1]
    ids = data[2]

    val_workers = int(4)

    train_inds, val_inds, test_inds = k_fold_split_train_val_test(len(inputs), fold_num= 2)
    test_dataset = neckNavigatorDataset(inputs = inputs, targets = targets, image_inds = test_inds)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    
    model_dir = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1/"
    testdataloader_dir = "/home/olivia/Documents/Internship_sarcopenia/locating_c3/attempt1/test_dataloader.pt"
    device = 'cuda:1'

    tester = neckNavigatorTest2(model_dir, test_dataloader, device)
    #test_results = tester
    C3s, segments, GTs = tester
    
    print("gt info: ", len(GTs))
    print(GTs.shape,)
    print("segs info: ", segments.shape)
    #print(segments[0].shape, len(segments))
    #print(C3s[0].shape)
    #c3 = C3s[0][0]
    #segment = segments[0][0]

    difference = euclid_dis(GTs, segments)
    average_distance = np.average(difference)
    print("average difference: ",average_distance)
    print("distance: ", difference)
    PrintSlice(C3s[0], segments[0], show=True)
    for j in range(0,4):
        projections(C3s[j],segments[j], order = [1,2,0])

    return

if __name__ == '__main__':
    main()