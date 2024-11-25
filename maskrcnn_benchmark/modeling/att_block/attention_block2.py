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
        self.conv2_layers = []
        for idx in range(5):
            conv1_block = "da_img_conv1_level{}".format(idx)
            conv2_block = "da_img_conv2_level{}".format(idx)
            conv1_block_module = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
            conv2_block_module = nn.Conv2d(512, 1, kernel_size=3, stride=1,padding=1)
            for module in [conv1_block_module, conv2_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 0)
            self.add_module(conv1_block, conv1_block_module)
            self.add_module(conv2_block, conv2_block_module)
            self.conv1_layers.append(conv1_block)
            self.conv2_layers.append(conv2_block)


    def forward(self, x1, x2):
        img_features = []
        
        for feature1, feature2, conv1_block, conv2_block in zip(x1, x2, self.conv1_layers, self.conv2_layers):
            #feature = torch.cat([feature1,feature2],1)
            feature = feature1-feature2
            inner_lateral = getattr(self, conv1_block)(feature)
            last_inner = F.relu(inner_lateral)
            att_scores=F.sigmoid(getattr(self, conv2_block)(last_inner))     
            img_features.append(feature1+att_scores*feature2) # (1.0+att_scores)*feature1+(2.0-att_scores)*feature2)
            
        return img_features
      
def build_attblock(cfg):
    if cfg.DATASETS.TOGETHER:
      in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
      return Attblock(in_channels)
    return []
    