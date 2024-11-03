import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class resnext_residualbottleneck_block(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 stride=1,
                 groups=32,
                 activation=torch.nn.ReLU
                 ):
        super().__init__()

        self.stride = stride
        self.channels_per_groups = max(1, int(inner_channels / groups))

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=inner_channels,
                            kernel_size=1,
                            stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(inner_channels),                  ##ex:128
            activation(),
            torch.nn.Conv2d(inner_channels,
                            inner_channels,
                            kernel_size=3,
                            stride=self.stride,
                            padding=1,
                            bias=False,
                            groups=self.channels_per_groups),
            torch.nn.BatchNorm2d(inner_channels),
            activation(),
            torch.nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),

        )

        self.final_activation = activation()
        self.down_skip_connection = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=self.stride)
        self.dim_equalizer = torch.nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)

    def forward(self, x):
        if self.stride == 2:
            down = self.down_skip_connection(x)
            out = self.features(x)
            out = out + down

        else:
            out = self.features(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        out = self.final_activation(out)
        return out