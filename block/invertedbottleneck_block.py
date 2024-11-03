import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class invertedbottleneck_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=4, stride=1, activation=torch.nn.ReLU6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_rate = expansion_rate
        self.stride = stride

        self.expansion_out = int(self.in_channels * self.expansion_rate)

        self.conv_expansion = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                  in_channels=self.in_channels,
                                                                  out_channels=self.expansion_out,
                                                                  bias=False,
                                                                  stride=1),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_depthwise = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=3,
                                                                  in_channels=self.expansion_out,
                                                                  out_channels=self.expansion_out,
                                                                  groups=self.expansion_out,
                                                                  bias=False,
                                                                  padding=1,
                                                                  stride=self.stride),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_projection = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                   in_channels=self.expansion_out,
                                                                   out_channels=self.out_channels,
                                                                   bias=False),
                                                   torch.nn.BatchNorm2d(self.out_channels))


    def forward(self, x):
        out = self.conv_expansion(x)
        out = self.conv_depthwise(out)
        out = self.conv_projection(out)

        if self.stride != 2 and self.in_channels == self.out_channels:
            out = out + x

        return out