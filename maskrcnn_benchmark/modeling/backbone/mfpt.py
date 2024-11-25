
from collections import OrderedDict
from locale import normalize

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import fpt4 as fpt4_module
from . import fpt3 as fpt3_module
from . import cfpt as cfpt_module
from . import resnet
import pdb



def build_mfpt(cfg):
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

    if cfg.MODEL.NUM_MFPT == 1:
        if  len(cfg.MODEL.RPN.ANCHOR_STRIDE)==3 and cfg.MODEL.BACKBONE.LASTLEVEL ==False:
            fpt1 = fpt3_module.FPT(out_channels,window_size = cfg.MODEL.WINDON_SIZE, share_fpt = cfg.MODEL.SHARE_FPT, position_emd=cfg.MODEL.POSITION_EMD, need_head=False, top_blocks =None, with_norm ="group_norm")
        elif len(cfg.MODEL.RPN.ANCHOR_STRIDE)==4 and cfg.MODEL.BACKBONE.LASTLEVEL ==True:
            fpt1 = fpt3_module.FPT(out_channels,window_size = cfg.MODEL.WINDON_SIZE, share_fpt = cfg.MODEL.SHARE_FPT, position_emd=cfg.MODEL.POSITION_EMD, need_head=False, top_blocks =None, with_norm ="group_norm")
        else:
            fpt1 = fpt4_module.FPT(out_channels,window_size = cfg.MODEL.WINDON_SIZE, share_fpt = cfg.MODEL.SHARE_FPT, position_emd=cfg.MODEL.POSITION_EMD, need_head=False, top_blocks =None, with_norm ="group_norm")
        if cfg.MODEL.BACKBONE.LASTLEVEL == True:
            cfpt1 = cfpt_module.CFPT(out_channels,window_size = cfg.MODEL.WINDON_SIZE, top_blocks=fpt4_module.LastLevelMaxPool(), with_norm ="group_norm")
        else:
            cfpt1 = cfpt_module.CFPT(out_channels,window_size = cfg.MODEL.WINDON_SIZE,top_blocks=None,with_norm ="group_norm")
        #model = nn.Sequential(OrderedDict([('cfpt1',cfpt1)]))
        if cfg.MODEL.USE_FPT == False:
            model = nn.Sequential(OrderedDict([('cfpt1',cfpt1)]))
        elif cfg.MODEL.USE_CFPT == False:
            model = nn.Sequential(OrderedDict([('fpt1',fpt1)]))
        else:
            model = nn.Sequential(OrderedDict([("fpt1", fpt1),('cfpt1',cfpt1)]))
            
    return model