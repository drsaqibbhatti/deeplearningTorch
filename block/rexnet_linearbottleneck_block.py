import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.se_conv_block import se_conv_block


class rexnet_linearbottleneck_block(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_se,
                 stride,
                 expand_rate=6,
                 se_rate=12):
        super(rexnet_linearbottleneck_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_channels = self.in_channels * expand_rate
        self.use_se = use_se
        self.stride = stride
        self.se_rate = se_rate

        self.use_skip = self.stride == 1 and self.in_channels <= self.out_channels

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=self.in_channels,
                            out_channels=self.expand_channels,
                            bias=False,
                            stride=1),
            torch.nn.BatchNorm2d(num_features=self.expand_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(kernel_size=3,
                            in_channels=self.expand_channels,
                            out_channels=self.expand_channels,
                            bias=False,
                            stride=self.stride,
                            groups=self.expand_channels,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=self.expand_channels)
        )

        if self.use_se == True:
            self.features.add_module('se_module',
                                     se_conv_block(in_channels=self.expand_channels,
                                                 channels=self.expand_channels,
                                                 se_rate=self.se_rate))
        self.features.add_module('relu6_layer', torch.nn.ReLU6())
        self.features.add_module('project_conv', torch.nn.Conv2d(kernel_size=1,
                                                                 in_channels=self.expand_channels,
                                                                 out_channels=self.out_channels,
                                                                 bias=False,
                                                                 stride=1))
        self.features.add_module('project_batch_norm', torch.nn.BatchNorm2d(num_features=self.out_channels))

    def forward(self, x):
        out = self.features(x)
        if self.use_skip == True:
            out[:, 0:self.in_channels] += x
        return out