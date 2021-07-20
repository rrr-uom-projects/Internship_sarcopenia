
import DUnet
from DUnet import UNet

model = UNet(in_channels = 1,
 out_channels = 2,
 n_blocks = 3,
 start_filts= 32,
 up_mode ='transpose',
 merge_mode = 'concat',
 planar_blocks = (),
 batch_norm='unset',
 attention= False,
 activation = 'relu',
 normalization= 'batch',
 full_norm= True,
 dim=3,
 conv_mode= 'same')