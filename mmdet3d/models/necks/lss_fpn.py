# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4,
                 input_feature_index = (0,2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
                      kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample , mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral=  lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral,
                          kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.extra_upsample:
            x = self.up2(x)
        return x



