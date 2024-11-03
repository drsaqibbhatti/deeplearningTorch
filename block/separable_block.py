import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class separable_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False):
        super(separable_block, self).__init__()

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

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out