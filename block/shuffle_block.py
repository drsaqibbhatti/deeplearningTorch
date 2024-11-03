import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class shuffle_block(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=3,
                 grouped_conv=True,
                 combine='add',
                 activation=torch.nn.ReLU):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        self.combine = combine

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self.out_channels -= self.in_channels
            self._combine_func = self._concat
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        if self.grouped_conv == True:
            self.first_1x1_groups = self.groups
        else:
            self.first_1x1_groups = 1

        self.group_compress_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=self.in_channels,
                            out_channels=self.bottleneck_channels,
                            groups=self.first_1x1_groups,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=self.bottleneck_channels),
            activation()
        )

        self.depthwise_conv = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3,
                            in_channels=self.bottleneck_channels,
                            out_channels=self.bottleneck_channels,
                            stride=self.depthwise_stride,
                            padding=1,
                            groups=self.bottleneck_channels,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=self.bottleneck_channels)
        )

        self.group_expand_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=self.bottleneck_channels,
                            out_channels=self.out_channels,
                            groups=self.groups,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=self.out_channels)
        )


    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out


    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):

        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)

        output = self.group_compress_convolution(x)
        output = channel_shuffle(output, groups=self.groups)
        output = self.depthwise_conv(output)
        output = self.group_expand_convolution(output)
        output = self._combine_func(residual, output)

        return F.relu(output)