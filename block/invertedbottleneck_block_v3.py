import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.hardswish_se_block import hardswish_se_block

class invertedbottleneck_block_v3(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 expansion_out,
                 out_channels,
                 stride=1,
                 kernel_size=3,
                 use_se=False,
                 activation=torch.nn.ReLU6):
        super().__init__()

        self.in_channels = in_channels
        self.stride = stride
        self.use_se = use_se
        self.expansion_out = expansion_out
        self.out_channels = out_channels
        self.padding = (kernel_size - 1) // 2

        self.conv_expansion = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                  in_channels=self.in_channels,
                                                                  out_channels=self.expansion_out,
                                                                  bias=False,
                                                                  stride=1),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_depthwise = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=kernel_size,
                                                                  in_channels=self.expansion_out,
                                                                  out_channels=self.expansion_out,
                                                                  groups=self.expansion_out,
                                                                  bias=False,
                                                                  padding=self.padding,
                                                                  stride=self.stride),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.squeeze_layer = hardswish_se_block(in_channels=self.expansion_out,
                                              reduction_ratio=4)

        self.conv_projection = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                   in_channels=self.expansion_out,
                                                                   out_channels=self.out_channels,
                                                                   bias=False),
                                                   torch.nn.BatchNorm2d(self.out_channels))


    def forward(self, x):
        out = self.conv_expansion(x)
        out = self.conv_depthwise(out)

        if self.use_se is True:
            out = self.squeeze_layer(out) * out

        out = self.conv_projection(out)

        if self.stride != 2 and self.in_channels == self.out_channels:
            out = out + x

        return out