#last edit Hermione: 04/008/2021
#trying to add attention gating and deep supervision
#basically a file to mess up eds models and Ozan Oktay's models
#adapting ed head to be neck nav
#Hermione and Olivia
# code taken and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter/models.py with permission

#imports
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################################################
############################  Neck Navigator model ##################################
#####################################################################################
class UnetDsv3(nn.Module):
    #deep supervision
    #from: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/models/networks/utils.py#L454
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)

class neckNavigator(nn.Module):
    def __init__(self, filter_factor=2, targets=1, in_channels=3):
        super(neckNavigator, self).__init__()
        ff = filter_factor # filter factor (easy net scaling)
        #old Input --> (3, 48, 120, 120)
        #new inp --> (1,128,128,128)
        # conv layers set 1 - down 1
        self.c1 = nn.Conv3d(in_channels=in_channels, out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(int(16*ff))
        self.drop1 = nn.Dropout3d(p=0.5)
        self.c2 = nn.Conv3d(in_channels=int(16*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(int(32*ff))
        self.drop2 = nn.Dropout3d(p=0.5)
        # conv layers set 2 - down 2
        self.c3 = nn.Conv3d(in_channels=int(32*ff)+in_channels, out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(int(32*ff))
        self.drop3 = nn.Dropout3d(p=0.5)
        self.c4 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(int(64*ff))
        self.drop4 = nn.Dropout3d(p=0.5)
        # conv layers set 3 - base
        self.c5 = nn.Conv3d(in_channels=int(64*ff)+in_channels, out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(int(64*ff))
        self.drop5 = nn.Dropout3d(p=0.5)
        self.c6 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(int(64*ff))
        self.drop6 = nn.Dropout3d(p=0.5)
        # upsample 1
        self.rc_1 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn_r1 = nn.BatchNorm3d(int(64*ff))
        self.drop_r1 = nn.Dropout3d(p=0.5)
        # conv layer set 4 - up 1
        self.c7 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(int(32*ff))
        self.drop7 = nn.Dropout3d(p=0.5)
        self.c8 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(int(32*ff))
        self.drop8 = nn.Dropout3d(p=0.5)
        # upsample 2
        self.rc_2 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn_r2 = nn.BatchNorm3d(int(32*ff))
        self.drop_r2 = nn.Dropout3d(p=0.5)
        # conv layer set 5 - up 2
        self.c9 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm3d(int(16*ff))
        self.drop9 = nn.Dropout3d(p=0.5)
        self.c10 = nn.Conv3d(in_channels=int(16*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm3d(int(16*ff))
        self.drop10 = nn.Dropout3d(p=0.5)
        # deep supervision
        self.dsv2 = UnetDsv3(in_size=int(32*ff), out_size=int(targets), scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=int(16*ff), out_channels=int(targets), kernel_size=1)
        # prediction convolution
        self.pred = nn.Conv3d(in_channels=int(16*ff), out_channels=int(targets), kernel_size=1)

        
        # final conv (without any concat)
        self.final = nn.Conv3d(int(targets)*2, int(targets), 1)


    @torch.cuda.amp.autocast()
    def forward(self, im):
        # Down block 1
        x = F.relu(self.bn1(self.c1(im)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.c2(x)))
        skip1 = self.drop2(x)
        x = F.max_pool3d(skip1, (2,2,2))
        downscaled_im = F.interpolate(im, scale_factor=0.5, mode='trilinear', align_corners=False)
        x = torch.cat((x, downscaled_im), dim=1)

        # Down block 2
        x = F.relu(self.bn3(self.c3(x)))
        x = self.drop3(x)
        x = F.relu(self.bn4(self.c4(x)))
        skip2 = self.drop4(x)
        x = F.max_pool3d(skip2, (2,2,2))
        downscaled_im = F.interpolate(downscaled_im, scale_factor=0.5, mode='trilinear', align_corners=False)
        x = torch.cat((x, downscaled_im), dim=1)

        # Base block 
        x = F.relu(self.bn5(self.c5(x)))
        x = self.drop5(x)
        x = F.relu(self.bn6(self.c6(x)))
        x = self.drop6(x)

        # Upsample 1
        x = F.relu(self.bn_r1(self.rc_1(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))))
        x = self.drop_r1(x)
        #attention here

        # Up block 1
        x = F.relu(self.bn7(self.c7(x+skip2)))
        x = self.drop7(x)
        x = F.relu(self.bn8(self.c8(x)))
        x = self.drop8(x)
        up2 = x

        # Upsample 2
        x = F.relu(self.bn_r2(self.rc_2(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))))
        x = self.drop_r2(x)

        # Up block 2
        x = F.relu(self.bn9(self.c9(x+skip1)))
        x = self.drop9(x)
        x = F.relu(self.bn10(self.c10(x)))
        x = self.drop10(x)
        up1 = x
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2], dim=1))

        return final

        # Predict
        #return self.pred(x)

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])

class neckNavigator_multi_dsv(nn.Module):
    #make ed's multihead a layer deeper and add the deep supervision
    def __init__(self, in_channels =1, filter_factor=2): 
        super(neckNavigator_multi_dsv, self).__init__()
        ff = filter_factor # filter factor (easy net scaling)
        # Input --> (3, 48, 120, 120)
        # conv layers set 1 - down 1
        self.c1 = nn.Conv3d(in_channels=in_channels, out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(int(16*ff))
        self.drop1 = nn.Dropout3d(p=0.5)
        self.c2 = nn.Conv3d(in_channels=int(16*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(int(32*ff))
        self.drop2 = nn.Dropout3d(p=0.5)
        # conv layers set 2 - down 2
        self.c3 = nn.Conv3d(in_channels=int(32*ff)+in_channels, out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(int(32*ff))
        self.drop3 = nn.Dropout3d(p=0.5)
        self.c4 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(int(64*ff))
        self.drop4 = nn.Dropout3d(p=0.5)
        # conv layers set 3 - base
        self.c5 = nn.Conv3d(in_channels=int(64*ff)+in_channels, out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(int(64*ff))
        self.drop5 = nn.Dropout3d(p=0.5)
        self.c6 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(int(64*ff))
        self.drop6 = nn.Dropout3d(p=0.5)
        # upsample 1
        self.rc_1 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn_r1 = nn.BatchNorm3d(int(64*ff))
        self.drop_r1 = nn.Dropout3d(p=0.5)
        # conv layer set 4 - up 1
        self.c7 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(int(32*ff))
        self.drop7 = nn.Dropout3d(p=0.5)
        self.c8 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(int(32*ff))
        self.drop8 = nn.Dropout3d(p=0.5)
        # upsample 2
        # head a
        self.rc_2_a = nn.Conv3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn_r2_a = nn.BatchNorm3d(int(32*ff))
        self.drop_r2_a = nn.Dropout3d(p=0.5)
        # head b
        self.rc_2_b = nn.Conv3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn_r2_b = nn.BatchNorm3d(int(32*ff))
        self.drop_r2_b = nn.Dropout3d(p=0.5)
        # conv layer set 5 - up 2
        # head a
        self.c9_a = nn.Conv3d(in_channels=int(32*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn9_a = nn.BatchNorm3d(int(16*ff))
        self.drop9_a = nn.Dropout3d(p=0.5)
        self.c10_a = nn.Conv3d(in_channels=int(16*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn10_a = nn.BatchNorm3d(int(16*ff))
        self.drop10_a = nn.Dropout3d(p=0.5)
        # head b
        self.c9_b = nn.Conv3d(in_channels=int(32*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn9_b = nn.BatchNorm3d(int(16*ff))
        self.drop9_b = nn.Dropout3d(p=0.5)
        self.c10_b = nn.Conv3d(in_channels=int(16*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn10_b = nn.BatchNorm3d(int(16*ff))
        self.drop10_b = nn.Dropout3d(p=0.5)
        # deep supervision
        self.dsv2 = UnetDsv3(in_size=int(32*ff), out_size=1, scale_factor=2)
        self.dsv1_a = nn.Conv3d(in_channels=int(16*ff), out_channels=1, kernel_size=1)
        self.dsv1_b = nn.Conv3d(in_channels=int(16*ff), out_channels=1, kernel_size=1)
        # final conv (without any concat)
        self.pred_a = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=1)
        self.pred_b = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=1)
        #prediction convolutions
        #self.pred_a = nn.Conv3d(in_channels=int(16*ff), out_channels=1, kernel_size=1)
        #self.pred_b = nn.Conv3d(in_channels=int(16*ff), out_channels=1, kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, im):
        # Down block 1
        x = F.relu(self.bn1(self.c1(im)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.c2(x)))
        skip1 = self.drop2(x)
        x = F.max_pool3d(skip1, (2,2,2))
        downscaled_im = F.interpolate(im, scale_factor=0.5, mode='trilinear', align_corners=False)
        x = torch.cat((x, downscaled_im), dim=1)

        # Down block 2
        x = F.relu(self.bn3(self.c3(x)))
        x = self.drop3(x)
        x = F.relu(self.bn4(self.c4(x)))
        skip2 = self.drop4(x)
        x = F.max_pool3d(skip2, (2,2,2))
        downscaled_im = F.interpolate(downscaled_im, scale_factor=0.5, mode='trilinear', align_corners=False)
        x = torch.cat((x, downscaled_im), dim=1)

        # Base block 
        x = F.relu(self.bn5(self.c5(x)))
        x = self.drop5(x)
        x = F.relu(self.bn6(self.c6(x)))
        x = self.drop6(x)

        # Upsample 1
        x = F.relu(self.bn_r1(self.rc_1(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))))
        x = self.drop_r1(x)

        # Up block 1
        x = F.relu(self.bn7(self.c7(x+skip2)))
        x = self.drop7(x)
        x = F.relu(self.bn8(self.c8(x)))
        x = self.drop8(x)
        up1 = x

        # Upsample 2a
        x_a = F.relu(self.bn_r2_a(self.rc_2_a(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))))
        x_a = self.drop_r2_a(x_a)

        # Upsample 2b
        x_b = F.relu(self.bn_r2_b(self.rc_2_b(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))))
        x_b = self.drop_r2_b(x_b)

        # Up block 2a
        x_a = F.relu(self.bn9_a(self.c9_a(x_a+skip1)))
        x_a = self.drop9_a(x_a)
        x_a = F.relu(self.bn10_a(self.c10_a(x_a)))
        x_a = self.drop10_a(x_a)
        up2_a = x_a

        # Up block 2b
        x_b = F.relu(self.bn9_b(self.c9_b(x_b+skip1)))
        x_b = self.drop9_b(x_b)
        x_b = F.relu(self.bn10_b(self.c10_b(x_b)))
        x_b = self.drop10_b(x_b)
        up2_b = x_b

        #deep supervision
        dsv2 = self.dsv2(up1)
        dsv1_a = self.dsv1_a(up2_a)
        dsv1_b = self.dsv1_b(up2_b)
        # Predictions
        pred = torch.cat([self.pred_a(torch.cat([dsv2, dsv1_a], dim = 1)),self.pred_b(torch.cat([dsv2, dsv1_b], dim =1))], dim =1)
        return pred

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])