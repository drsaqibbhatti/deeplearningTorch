import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class residual_block(torch.nn.Module):

    def __init__(self, in_channels, inner_channels, out_channels, stride=1, activation=torch.nn.SiLU):
        super(residual_block, self).__init__()

        self.stride = stride
        self.features = torch.nn.Sequential(torch.nn.Conv2d(in_channels,
                                                            inner_channels,
                                                            kernel_size=3,
                                                            stride=self.stride,
                                                            padding=1,
                                                            bias=False),
                                            torch.nn.BatchNorm2d(num_features=inner_channels),
                                            activation(),
                                            torch.nn.Conv2d(inner_channels,
                                                            out_channels,
                                                            kernel_size=3,
                                                            padding='same',
                                                            bias=False),
                                            torch.nn.BatchNorm2d(num_features=out_channels))

        self.down_skip_connection = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=self.stride)
        self.dim_equalizer = torch.nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.final_activation = activation()



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