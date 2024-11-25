# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torch import nn
import torch
from maskrcnn_benchmark.structures.image_list import to_image_list
import os
import errno
import cv2
import numpy as np
import torch.nn.functional as F

from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads
from ..att_block.attention_block import build_attblock
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from ..backbone import build_mfpt, build_prefeature


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def compute_SMap(feat_maps, dest_layer):
    out = feat_maps[dest_layer]  # [bs, C, H, W]
    smaps = out.mean(dim=1)
    smaps = smaps.cpu().data.numpy()
    re = []
    for smap in smaps:
        smap -= smap.min()
        smap /= smap.max()
        re.append(smap)
    return re


def save_debug_images(batch_image, target, filepath):
    for i in range(batch_image.shape[0]):
        filename = "{}_{}.jpg".format(filepath, str(i))
        image = batch_image[i].permute(1, 2, 0).cpu().detach().numpy()  # CHW->HWC
        image -= image.min()
        image /= image.max()
        image = np.uint8(255 * image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        co = target[i].bbox
        dst = []
        for j in range(co.shape[0]):
            cood = [
                [int(co[j][0]), int(co[j][1])],
                [int(co[j][0]), int(co[j][3])],
                [int(co[j][2]), int(co[j][3])],
                [int(co[j][2]), int(co[j][1])],
            ]
            dst.append(cood)
        image = cv2.polylines(image, np.int32(dst), True, (240, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(filename, image)


class GradientNet(nn.Module):
    def __init__(self, device):
        super(GradientNet, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


def cal_target_mask(bbox, img_size, f_size, device):
    bs = len(bbox)
    target_mask = torch.zeros(bs, f_size[0], f_size[1]).to(device)
    k = img_size[0] / f_size[1]
    for i in range(bs):
        bbox_co = bbox[i].bbox
        for box in bbox_co:
            e_w = (box[2] / k - box[0] / k) / 2
            e_h = (box[3] / k - box[1] / k) / 2
            for x in range(int(box[0] / k - e_w), int(box[2] / k + e_w)):
                for y in range(int(box[1] / k - e_h), int(box[3] / k + e_h)):
                    target_mask[i][y][x] = 1
    return target_mask  # bs*h*w


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        if cfg.MODEL.RETINANET_ON:
            from ..rpn1.rpn import build_rpn
        else:
            from ..rpn2.rpn import build_rpn
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)
        self.att_blocks = build_attblock(cfg)
        self.mfpt = build_mfpt(cfg)
        self.save_featuremap = cfg.MODEL.SAVE_FEATURE
        self.save_debug_image = cfg.MODEL.SAVE_DEBUG_IMAGE
        self.output_dir = cfg.OUTPUT_DIR
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.f_selector = build_prefeature(cfg)

    def forward(self, images1, images2, targets1=None, targets2=None, iteration=1):
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
        if self.training and targets1 is None:
            raise ValueError("In training mode, targets should be passed")

        images = torch.cat((images1.tensors, images2.tensors), 0)  # (B*2)*C*W*H, without selector
        # images = torch.cat((images1_, images2_), 0)  # (B*2)*C*W*H, with selector
        features = self.backbone(images)
        l = int(features[0].shape[0] / 2)
        # pdb.set_trace()

        if self.save_featuremap:
            SMap_before1 = [[] for i in range(0, l)]
            SMap_before2 = [[] for i in range(0, l)]
            for i_ in range(0, len(features)):
                t_ = compute_SMap(features, i_)
                for j_ in range(0, l):
                    # pdb.set_trace()
                    SMap_before1[j_].append(t_[j_])
                    SMap_before2[j_].append(t_[j_ + l])


        features1 = []
        features2 = []
        for i in range(0, len(features)):
            features1.append(features[i][0:l])
            features2.append(features[i][l:])

        loss_t, loss_rgb, features1_, features2_ = self.select_feature(features1, features2, targets1)  # rgb', t'

        if self.save_featuremap:
            SMap_after1 = [[] for i in range(0, l)]
            SMap_after2 = [[] for i in range(0, l)]
            for i_ in range(0, len(features1_)):
                t_ = compute_SMap(features1_, i_)
                t__ = compute_SMap(features2_, i_)
                for j_ in range(0, l):
                    SMap_after1[j_].append(t_[j_])
                    SMap_after2[j_].append(t__[j_])

        for i in range(len(features1)):
            features1[i] = torch.cat((features1[i], features1_[i]), 1)
            features2[i] = torch.cat((features2[i], features2_[i]), 1)
        # pdb.set_trace()
        proposals1, proposal_losses1 = self.rpn(images1, features1, targets1)
        proposals2, proposal_losses2 = self.rpn(images2, features2, targets2)

        da_losses = {}
        if self.roi_heads:
            # pdb.set_trace()
            # x, result, detector_losses, da_ins_feas, da_ins_labels, da_proposals = self.roi_heads(features, proposals, targets)

            x, result1, detector_losses1, da_ins_feas1, da_ins_labels1, da_proposals1 = self.roi_heads(features1,
                                                                                                       proposals1,
                                                                                                       targets1)
            x, result2, detector_losses2, da_ins_feas2, da_ins_labels2, da_proposals2 = self.roi_heads(features2,
                                                                                                       proposals2,
                                                                                                       targets2)

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
            losses1.update(loss_rgb)
            losses1.update(loss_t)
            losses2 = {}
            # losses2.update(lossrgb)
            losses2.update(detector_losses2)
            losses2.update(proposal_losses2)


            return losses1, losses2

        if self.save_featuremap:
            return result1, result2, SMap_before1, SMap_before2, SMap_after1, SMap_after2
        return result1, result2

    def select_feature(self, features1, features2, targets1):
        loss_t = {'loss_t': 0}
        loss_rgb = {'loss_rgb': 0}
        gradient_model = GradientNet(self.device).to(self.device)
        f1, f2 = self.f_selector(features1, features2)

        for i in range(len(features1)):
            loss_t['loss_t'] += self.TLoss(f1[i], features1[i], features2[i], targets1, self.device)
            loss_rgb['loss_rgb'] += self.RGBLoss(f2[i], features1[i], features2[i], gradient_model)

        loss_t['loss_t'] /= len(features1)
        loss_rgb['loss_rgb'] /= len(features1)
        return loss_t, loss_rgb, f2, f1

    def TLoss(self, fused_feature, t_feature, rgb_feature, bbox_t, device):
        if bbox_t is None:
            return 0
        bs, h, w = fused_feature.shape[0], fused_feature.shape[2], fused_feature.shape[3]
        fused_feature = fused_feature.mean(dim=1)
        t_feature = t_feature.mean(dim=1)
        return abs(fused_feature.sum() - t_feature.sum()) / (bs * w * h)

    def RGBLoss(self, fused_feature, t_feature, rgb_feature, gradient_model):
        loss = 0
        bs, c, w, h = fused_feature.shape
        for i in range(bs):
            for j in range(c):
                f_f = torch.reshape(fused_feature[i][j], (1, 1, w, h))
                t_f = torch.reshape(t_feature[i][j], (1, 1, w, h))
                rgb_f = torch.reshape(rgb_feature[i][j], (1, 1, w, h))
                g_fused = gradient_model(f_f)
                g_rgb = gradient_model(rgb_f)
                loss += abs(g_fused.sum() - g_rgb.sum())
        return loss / (bs * c * w * h)
