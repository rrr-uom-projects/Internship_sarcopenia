# 27/07/2021 added to inference: 09/09/2021
# Hermione Warr and Olivia Murray
# Neck Navigator Model, maybe also put muscle mapper here too - not much to add. 
# Code taken and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter/models.py with permission

#imports
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


######***\~ NECK NAVIGATOR MODEL ~/***######

class neckNavigator(nn.Module):
    def __init__(self, filter_factor=2, targets=1, in_channels=1):
        super(neckNavigator, self).__init__()
        ff = filter_factor # filter factor (easy net scaling)
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
        # prediction convolution
        self.pred = nn.Conv3d(in_channels=int(16*ff), out_channels=int(targets), kernel_size=1)
        #activation
        self.act = nn.Sigmoid()

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

        # Upsample 2
        x = F.relu(self.bn_r2(self.rc_2(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False))))
        x = self.drop_r2(x)

        # Up block 2
        x = F.relu(self.bn9(self.c9(x+skip1)))
        x = self.drop9(x)
        x = F.relu(self.bn10(self.c10(x)))
        x = self.drop10(x)
        x = self.pred(x)
        #x = self.act(x) #only need if MSE loss
        return x

    def load_previous(self, checkpoint_dir, logger):
        # load last checkpoint weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'last_checkpoint.pytorch'))
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