import random
from sklearn.model_selection import KFold

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
