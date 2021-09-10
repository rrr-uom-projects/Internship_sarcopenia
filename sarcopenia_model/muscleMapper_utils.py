import random
from sklearn.model_selection import KFold
import numpy as np

def k_fold_cross_val(dataset_size, num_splits):
    train =[]
    test = []
    np.random.seed(2305)
    #shuffled_ind_list = np.random.permutation(dataset_size)
    shuffled_ind_list = np.arange(dataset_size)
    #print(shuffled_ind_list)
    kf = KFold(n_splits = num_splits, shuffle = True, random_state=np.random.seed(2305))
    for train_index, test_index in kf.split(shuffled_ind_list):
          print("TRAIN:", train_index, "\nTEST:", test_index)
          train.append(train_index)
          test.append(test_index)
    return train, test
def dataset_TVTsplit(inputs, targets, train_inds, val_inds, test_inds):
    def select_data(im_inds):
        selected_im = [inputs[ind] for ind in im_inds]
        selected_masks = [targets[ind] for ind in im_inds]
        return selected_im, selected_masks

    train_inputs, train_masks = select_data(train_inds)
    val_inputs, val_masks = select_data(val_inds)
    test_inputs, test_masks = select_data(test_inds)

    return train_inputs, train_masks, val_inputs, val_masks, test_inputs, test_masks
