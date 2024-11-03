import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.se_block import se_block

class se_separable_act_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False, activation=torch.nn.SiLU):
        super(se_separable_act_block, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         groups=in_channels,
                                         bias=bias,
                                         stride=stride,
                                         padding=padding)

        self.pointwise = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         bias=bias)

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.activation1 = activation()

        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.activation2 = activation()

        self.seblock = se_block(in_channels=out_channels, activation=activation)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.activation2(out)
        out = self.seblock(out) * out
        return out