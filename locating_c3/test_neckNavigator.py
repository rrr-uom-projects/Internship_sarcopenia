#created by hermione on 16/08/2021
#to test the model at various stages

from utils import PrintSlice, get_data, projections, euclid_dis, display_net_test, slice_preds, GetSliceNumber
from neckNavigatorTester import neckNavigatorTest2
from neckNavigatorTester import neckNavigatorTest1

from neckNavigator import neckNavigator
from neckNavigatorData import neckNavigatorDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from neckNavigatorUtils import k_fold_split_train_val_test
import pandas as pd 


def mrofsnart(net_slice, transforms, shape = 128, coords = None, test_inds = None):#transforms backwards
    #might have get transform indices for test data
    if test_inds is not None:
        transforms = [transforms[ind] for ind in test_inds]
    x_arr,y_arr,z_arr = [],[],[]
    for i in range(len(net_slice)):
        #undo scale
        net_slice[i] *= 14/16

        #print(net_slice[i], transforms[i][1][0])

        #undo crop
        #eg z crop [46,1] z=12  [[true, crop array]<- crop[zmin, zmax, xmin,...],]
        z = net_slice[i] + transforms[i][1][0]
        if coords is not None:
            x = coords[i,0]*2
            y = coords[i,1]*2
            x += transforms[i][1][2]
            y += transforms[i][1][4]
            x_arr.append(x)
        #undo flip if necessary
            if (transforms[i][0]==True):
                y = shape - y
            y_arr.append(y)
        if (transforms[i][0]==True):
            z = shape - z

        #print(z)

        z_arr.append(z)
    return np.array(x_arr),np.array(y_arr),np.array(z_arr)

def main():

    # get data
    #data_path =  '/home/olivia/Documents/Internship_sarcopenia/locating_c3/preprocessed_Tgauss.npz'
    data_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/preprocessed_Tgauss.npz'
    save_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/fold_info/c3_loc.xlsx'#' +str(i) + '
    xl_writer = pd.ExcelWriter(save_path)

    data = get_data(data_path)
    inputs = data[0]
    targets = data[1]
    ids = data[2]
    transforms = data[3]
    org_slices = data[4]

    val_workers = int(4)

    train_inds, val_inds, test_inds = k_fold_split_train_val_test(len(inputs), fold_num= 2)

    test_dataset = neckNavigatorDataset(inputs = inputs, targets = targets, im_inds = test_inds)

    test_dataloader = DataLoader(dataset= test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    print(test_inds)
    model_dir = "/home/hermione/Documents/Internship_sarcopenia/locating_c3/model_ouputs"
    
    device = 'cuda:1'

    tester = neckNavigatorTest2(model_dir, test_dataloader, device)

    #test_results = tester
    C3s, segments, GTs = tester

    difference = euclid_dis(GTs, segments)
    #print(difference)
    #projections(C3s[3],segments[3], order = [2,1,0], show=True)
    #projections(C3s[1],GTs[1], order = [1,2,0], show=True)
    slice_preds_q = display_net_test(C3s, segments, GTs, ids)

    slice_no_preds = slice_preds(segments)
    slice_no_gts = slice_preds(GTs)
    slice_no_gts_test = []

    for i in range(len(GTs)): #sanity check
        slice_no_gt = GetSliceNumber(GTs[i])
        slice_no_gts_test.append(slice_no_gt)
        #if slice_no_gt != slice_no_gts[i]: print("well shit")

    print("Net Preds: ",slice_no_preds.shape)
    #print("GTS: ", slice_no_gts)
    #print("checking...", slice_no_gts_test)

    #c3_loc_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/c3_loc.npz'
    #np.savez(c3_loc_path, inputs = inputs, masks = segments, slice_nos = slice_no_preds, ids = ids)
    test_ids = [ids[ind] for ind in test_inds]
    test_org_slices = [org_slices[ind] for ind in test_inds]
  
    #undoing transforms to get corect slice number
    x,y,z = mrofsnart(slice_no_preds, transforms, test_inds =test_inds)
    print("length: ", len(z), len(test_ids), len(slice_no_preds), len(test_org_slices) )
    df = pd.DataFrame({"IDs": test_ids, "Slice_Numbers": slice_no_preds, "PostProcess Slice numbers": z, "GT org slices": test_org_slices, "GT T Slices": slice_no_gts})
    save_path = '/home/hermione/Documents/Internship_sarcopenia/locating_c3/c3_loc.xlsx'
    df.to_excel(excel_writer = save_path, index=False,
             sheet_name="data")
    print("Saved predictions")

     #####*** POST-PROCESSING ***#####
    #     slice_no_preds = slice_preds(segments)  
    #     slice_no_gts = slice_preds(GTs)
    #     x,y,z = mrofsnart(slice_no_preds, transforms, test_inds = test_array[i])
    #     _,_,z_test  = mrofsnart(slice_no_gts, transforms, test_inds = test_array[i])

    #     difference_old = euclid_dis(GTs, segments)
    #     difference, mm_distance= euclid_diff_mm(test_org_slices, z, test_vox_dims)
       
    #     ###*** SAVING TEST INFO ***###
    #     df = pd.DataFrame({"IDs": ids[test_array[i]], "Out_Slice_Numbers": slice_no_preds, "GT_ProcessedSliceNo": slice_no_gts, "PostProcessSliceNo": z, 
    #     "GT_Org_Slice_No": test_org_slices, "GT_z_test": z_test, "SliceDifferences": difference,"OldSliceDifferences": difference_old, "z_distance": mm_distance}) 
    #         #"y_distance": mm_threeD_distance[1], "z_disance":mm_threeD_distance[2], "pythag_dist_abs": pythagoras_dist})
    #     df.to_excel(excel_writer = xl_writer, index=False,
    #             sheet_name=f'fold{i+1}')
    #     xl_writer.save()
        
    # xl_writer.close()


    return

if __name__ == '__main__':
    main()