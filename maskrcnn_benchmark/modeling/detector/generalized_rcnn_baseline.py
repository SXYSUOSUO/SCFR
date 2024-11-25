# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torch import nn
import torch
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads
# from ..da_headc.da_heads import build_da_heads
from ..att_block.attention_block import build_attblock
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from ..backbone import build_mfpt

import pdb

def compute_SMap(feat_maps,dest_layer):
    out = feat_maps[dest_layer]  # [bs, C, H, W]
    smaps = out.mean(dim=1)
    smaps = smaps.cpu().data.numpy()
    re = []
    for i, smap in enumerate(smaps):
        smap -= smap.min()
        smap /= smap.max()
        #heatmap = cv2.resize(smap, (112, 112))
        #heatmap = np.uint8(255 * heatmap)
        #cv2.imwrite(fpath+'{}_{}.jpg'.format(i, dest_layer), heatmap)
        re.append(smap)
    return re


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)
        self.att_blocks = build_attblock(cfg)
        self.mfpt = build_mfpt(cfg)
        self.save_featuremap = cfg.MODEL.SAVE_FEATURE

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        features = self.backbone(images.tensors)
        # res_feat = self.backbone.body(images.tensors)
        # features = self.backbone.fpn(res_feat)

        proposals, proposal_losses = self.rpn(images, features, targets)
        da_losses = {}
        
        if self.roi_heads:
            #pdb.set_trace()
            x, result, detector_losses, da_ins_feas, da_ins_labels, da_proposals = self.roi_heads(features, proposals, targets)

            # import glob
            # all_mat = glob.glob('/home/yc/workplace/code/daf-dev/tmp_*.mat')
            # if len(all_mat) < 10:
            #     import scipy.io as sio
            #     sio.savemat('/home/yc/workplace/code/daf-dev/tmp_{}.mat'.format(len(all_mat)),
            #             {'labels': da_ins_labels.detach().cpu().numpy(),
            #             'ins_feat': da_ins_feas.detach().cpu().numpy(),
            #             'da_proposals_src': da_proposals[0].bbox.detach().cpu().numpy(),
            #             'da_proposals_tgt': da_proposals[1].bbox.detach().cpu().numpy()})
            #pdb.set_trace()
            if self.da_heads:
              da_losses = self.da_heads(result, features, da_ins_feas, da_ins_labels, da_proposals, targets)
              # da_losses = self.da_heads(result, res_feat, da_ins_feas, da_ins_labels, da_proposals, targets)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses

        return result


    def forward2(self, images1, images2, targets1=None, targets2=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets1  is None:
            raise ValueError("In training mode, targets should be passed")
        
        images1 = to_image_list(images1)
        features1 = self.backbone(images1.tensors)
        
        
        
        images2 = to_image_list(images2)
        features2 = self.backbone(images2.tensors)

        features= self.att_blocks(features1,features2)
        #print (att_scores[0][0][0][0])
        
        

        
        proposals1, proposal_losses1 = self.rpn(images1, features, targets1)  
        proposals2, proposal_losses2 = self.rpn(images2, features, targets2)


    
        #pdb.set_trace()
        da_losses = {}
        if self.roi_heads:
            #pdb.set_trace()
            #x, result, detector_losses, da_ins_feas, da_ins_labels, da_proposals = self.roi_heads(features, proposals, targets)




            x, result1, detector_losses1, da_ins_feas1, da_ins_labels1, da_proposals1 = self.roi_heads(features, proposals1, targets1)
            #result2 = result1
            x, result2, detector_losses2, da_ins_feas2, da_ins_labels2, da_proposals2 = self.roi_heads(features, proposals2, targets2)
            # import glob
            # all_mat = glob.glob('/home/yc/workplace/code/daf-dev/tmp_*.mat')
            # if len(all_mat) < 10:
            #     import scipy.io as sio
            #     sio.savemat('/home/yc/workplace/code/daf-dev/tmp_{}.mat'.format(len(all_mat)),
            #             {'labels': da_ins_labels.detach().cpu().numpy(),
            #             'ins_feat': da_ins_feas.detach().cpu().numpy(),
            #             'da_proposals_src': da_proposals[0].bbox.detach().cpu().numpy(),
            #             'da_proposals_tgt': da_proposals[1].bbox.detach().cpu().numpy()})

            if self.da_heads:
              da_losses = self.da_heads(result1, features1, da_ins_feas1, da_ins_labels1, da_proposals1, targets1)
              # da_losses = self.da_heads(result, res_feat, da_ins_feas, da_ins_labels, da_proposals, targets)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result1, result2 = proposals1, proposals2 
            detector_losses1 = {}
            detector_losses2 = {}
   

        if self.training:
            losses1 = {}
            losses1.update(detector_losses1)
            losses1.update(proposal_losses1)
            losses2={} 
            losses2.update(detector_losses2)
            losses2.update(proposal_losses2)     
            losses2.update(da_losses)
       

            return losses1, losses2
        
        return result1,result2
