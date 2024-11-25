from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn

import pdb
class Attblock(nn.Module):
    """
    
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(Attblock, self).__init__()

        self.conv1_layers = []
        for idx in range(5):
            conv1_block = "da_img_conv1_level{}".format(idx)
            conv1_block_module = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=3, stride=1, padding=1)
            for module in [conv1_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 0)
            self.add_module(conv1_block, conv1_block_module)
            self.conv1_layers.append(conv1_block)
        


    def forward(self, x1, x2):
        img_features = []
        
        for feature1, feature2, conv1_block in zip(x1, x2, self.conv1_layers):
            feature = torch.cat([feature1,feature2],1)
            feature = getattr(self, conv1_block)(feature)
            feature = F.relu(feature)
            img_features.append(feature)  #+(2.0-att_scores)*feature2)
        #print (att_scores[0][0][0][0])
            
        return img_features
      
def build_attblock(cfg):
    if cfg.DATASETS.TOGETHER:
      in_channels = 2*cfg.MODEL.BACKBONE.OUT_CHANNELS
      return Attblock(in_channels)
    return []
    