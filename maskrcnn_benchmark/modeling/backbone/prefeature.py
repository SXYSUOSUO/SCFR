from collections import OrderedDict
from locale import normalize

from torch import nn
import torch.nn.functional as F
import torch

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.structures.image_list import to_image_list

import pdb


class FeatureSelector(nn.Module):
    def __init__(self, cfg):
        super(FeatureSelector, self).__init__()
        channel = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.feature0 = nn.Sequential(
            nn.Conv2d(channel*3, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.ReLU(inplace=True)
                )
        self.feature1 = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.feature2 = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.feature3 = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.res0 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel * 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel * 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
                )
        self.res1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel * 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel * 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel* 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel * 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel * 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel * 2, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel * 2, momentum=0.1),
            nn.PReLU(),
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )

        self.feature0_t0 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature0_t1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature0_t2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )

        self.feature1_t0 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature1_t1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature1_t2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )

        self.feature2_t0 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature2_t1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature2_t2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )

        self.feature3_t0 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature3_t1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )
        self.feature3_t2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=(1, 1)),
            nn.BatchNorm2d(channel, momentum=0.1),
            nn.PReLU()
        )

    def forward(self, fea_t, fea_rgb):  # 4*[4, 256, w, h]
        # pdb.set_trace()
        feature0 = self.feature0(torch.cat((
            F.upsample(fea_rgb[1], size=fea_rgb[0].size()[2:], mode='bilinear', align_corners=True),
            F.upsample(fea_rgb[2], size=fea_rgb[0].size()[2:], mode='bilinear', align_corners=True),
            F.upsample(fea_rgb[3], size=fea_rgb[0].size()[2:], mode='bilinear', align_corners=True)
        ), 1))
        feature1 = self.feature1(torch.cat((
            F.interpolate(fea_rgb[0], size=fea_rgb[1].size()[2:], mode='bilinear', align_corners=True),
            F.upsample(fea_rgb[2], size=fea_rgb[1].size()[2:], mode='bilinear', align_corners=True),
            F.upsample(fea_rgb[3], size=fea_rgb[1].size()[2:], mode='bilinear', align_corners=True)
        ), 1))
        feature2 = self.feature2(torch.cat((
            F.interpolate(fea_rgb[0], size=fea_rgb[2].size()[2:], mode='bilinear', align_corners=True),
            F.interpolate(fea_rgb[1], size=fea_rgb[2].size()[2:], mode='bilinear', align_corners=True),
            F.upsample(fea_rgb[3], size=fea_rgb[2].size()[2:], mode='bilinear', align_corners=True)
        ), 1))
        feature3 = self.feature3(torch.cat((
            F.interpolate(fea_rgb[0], size=fea_rgb[3].size()[2:], mode='bilinear', align_corners=True),
            F.interpolate(fea_rgb[1], size=fea_rgb[3].size()[2:], mode='bilinear', align_corners=True),
            F.interpolate(fea_rgb[2], size=fea_rgb[3].size()[2:], mode='bilinear', align_corners=True)
        ), 1))

        rgb_feature0 = self.res0(torch.cat((feature0, fea_rgb[0]), 1)) + fea_rgb[0]
        rgb_feature1 = self.res1(torch.cat((feature1, fea_rgb[1]), 1)) + fea_rgb[1]
        rgb_feature2 = self.res2(torch.cat((feature2, fea_rgb[2]), 1)) + fea_rgb[2]
        rgb_feature3 = self.res3(torch.cat((feature3, fea_rgb[3]), 1)) + fea_rgb[3]

        rgb_ = [
            rgb_feature0,
            rgb_feature1,
            rgb_feature2,
            rgb_feature3
        ]

        feature0_t0 = self.feature0_t0(torch.cat((fea_t[0],
                                                  F.upsample(fea_t[1], size=fea_t[0].size()[2:], mode='bilinear',
                                                             align_corners=True)), 1))
        feature0_t1 = self.feature0_t1(torch.cat((feature0_t0,
                                                  F.upsample(fea_t[2], size=fea_t[0].size()[2:], mode='bilinear',
                                                             align_corners=True)), 1))
        feature0_t2 = self.feature0_t2(torch.cat((feature0_t1,
                                                  F.upsample(fea_t[3], size=fea_t[0].size()[2:], mode='bilinear',
                                                             align_corners=True)), 1))

        feature1_t0 = self.feature1_t0(torch.cat((fea_t[1],
                                                  F.upsample(fea_t[0], size=fea_t[1].size()[2:], mode='bilinear',
                                                             align_corners=True)), 1))
        feature1_t1 = self.feature1_t1(torch.cat((feature1_t0,
                                                  F.upsample(fea_t[2], size=fea_t[1].size()[2:], mode='bilinear',
                                                             align_corners=True)), 1))
        feature1_t2 = self.feature1_t2(torch.cat((feature1_t1,
                                                  F.interpolate(fea_t[3], size=fea_t[1].size()[2:], mode='bilinear',
                                                                align_corners=True)), 1))

        feature2_t0 = self.feature2_t0(torch.cat((fea_t[2],
                                                  F.upsample(fea_t[0], size=fea_t[2].size()[2:], mode='bilinear',
                                                             align_corners=True)), 1))
        feature2_t1 = self.feature2_t1(torch.cat((feature2_t0,
                                                  F.interpolate(fea_t[1], size=fea_t[2].size()[2:], mode='bilinear',
                                                                align_corners=True)), 1))
        feature2_t2 = self.feature2_t2(torch.cat((feature2_t1,
                                                  F.interpolate(fea_t[3], size=fea_t[2].size()[2:], mode='bilinear',
                                                                align_corners=True)), 1))

        feature3_t0 = self.feature2_t0(torch.cat((fea_t[3],
                                                  F.interpolate(fea_t[0], size=fea_t[3].size()[2:], mode='bilinear',
                                                                align_corners=True)), 1))
        feature3_t1 = self.feature2_t1(torch.cat((feature3_t0,
                                                  F.interpolate(fea_t[1], size=fea_t[3].size()[2:], mode='bilinear',
                                                                align_corners=True)), 1))
        feature3_t2 = self.feature2_t2(torch.cat((feature3_t1,
                                                  F.interpolate(fea_t[2], size=fea_t[3].size()[2:], mode='bilinear',
                                                                align_corners=True)), 1))


        t_ = [
            feature0_t2,
            feature1_t2,
            feature2_t2,
            feature3_t2
        ]

        return rgb_, t_ # t', rgb'


def build_prefeature(channel):
    model = FeatureSelector(channel)
    return model
