# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
from locale import normalize

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import fpn3 as fpn3_module
from . import cfpt as cfpt_module
from . import resnet
import pdb

@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS  
    if  len(cfg.MODEL.RPN.ANCHOR_STRIDE)==3:
         fpn = fpn3_module.FPN(
            in_channels_list=[
            in_channels_stage2 ,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,],
            out_channels=out_channels,
            conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU),
            #top_blocks=fpn_module.LastLevelMaxPool(),
            )
    else:
        fpn = fpn_module.FPN(
            in_channels_list=[
            in_channels_stage2 ,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,],
            out_channels=out_channels,
            conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU),
            #top_blocks=fpn_module.LastLevelMaxPool(),
            )
    #
    model = nn.Sequential(OrderedDict([("body", body),("fpn", fpn)]))
    #model = nn.Sequential(OrderedDict([("body", body)]))

  
    
    return model

@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        #top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels), # fpt_module.LastLevelP6P7(in_channels_p6p7, out_channels)
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    #fpt1 = fpt_module.FPT(out_channels,need_head=True,top_blocks =None, with_norm ="group_norm")
    #cfpt1 = cfpt_module.CFPT(out_channels,top_blocks=None,with_norm ="group_norm")
    #model = nn.Sequential(OrderedDict([("body", body), ("fpt1", fpt1)]))
    '''
    if cfg.MODEL.NUM_MFPT == 1:
        fpt1 = fpt_module.FPT(out_channels,need_head=True,top_blocks =None, with_norm ="group_norm")
        cfpt1 = cfpt_module.CFPT(out_channels,top_blocks=fpt_module.LastLevelMaxPool(),with_norm ="group_norm")
        model = nn.Sequential(OrderedDict([("body", body), ("fpt1", fpt1),('cfpt1',cfpt1)]))
    elif cfg.MODEL.NUM_MFPT == 2:
        fpt1 = fpt_module.FPT(out_channels,need_head=True,top_blocks =None, with_norm ="group_norm")
        cfpt1 = cfpt_module.CFPT(out_channels,top_blocks=None,with_norm ="group_norm")
        fpt2 = fpt_module.FPT(out_channels,need_head=False,top_blocks =fpt_module.LastLevelMaxPool(), with_norm ="group_norm")
        #cfpt2 = cfpt_module.CFPT(out_channels,top_blocks=None,with_norm ="layer_norm")
        model = nn.Sequential(OrderedDict([("body", body), ("fpt1", fpt1),('cfpt1',cfpt1)]))
    elif cfg.MODEL.NUM_MFPT == 3:
        fpt1 = fpt_module.FPT(out_channels,need_head=True,top_blocks =None, with_norm ="group_norm")
        cfpt1 = cfpt_module.CFPT(out_channels,top_blocks=None,with_norm ="group_norm")
        fpt2 = fpt_module.FPT(out_channels,need_head=False,top_blocks =None, with_norm ="group_norm")
        cfpt2 = cfpt_module.CFPT(out_channels,top_blocks=fpt_module.LastLevelMaxPool(),with_norm ="group_norm")
        #fpt3 = fpt_module.FPT(out_channels,need_head=False,top_blocks =None, with_norm ="layer_norm")
        #cfpt3 = cfpt_module.CFPT(out_channels,top_blocks=None,with_norm ="layer_norm")
        model = nn.Sequential(OrderedDict([("body", body), ("fpt1", fpt1),('cfpt1',cfpt1),("fpt2", fpt2),('cfpt2',cfpt2)]))
    '''

    return model
    #pdb.set_trace()
    #return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
