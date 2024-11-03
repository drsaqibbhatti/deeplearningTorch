import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class csp_residual_block(torch.nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, stride=1, part_ratio=0.5, activation=torch.nn.SiLU):
        super(csp_residual_block, self).__init__()

        self.part1_chnls = int(in_dim * part_ratio)
        self.part2_chnls = in_dim - self.part1_chnls                ##Residual Layer Channel Calculation

        self.part1_out_chnls = int(out_dim * part_ratio)
        self.part2_out_chnls = out_dim - self.part1_out_chnls

        self.stride = stride
        self.residual_block = torch.nn.Sequential(torch.nn.Conv2d(self.part2_chnls,
                                                                  mid_dim,
                                                                  kernel_size=3,
                                                                  stride=self.stride,
                                                                  padding=1,
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=mid_dim),
                                                  activation(),
                                                  torch.nn.Conv2d(mid_dim,
                                                                  self.part2_out_chnls,
                                                                  kernel_size=3,
                                                                  padding='same',
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=self.part2_out_chnls))

        self.projection1 = torch.nn.Conv2d(in_channels=self.part2_chnls,            ##Residual Projection
                                           out_channels=self.part2_out_chnls,
                                           kernel_size=1,
                                           stride=2)

        self.projection2 = torch.nn.Conv2d(in_channels=self.part1_chnls,
                                           out_channels=self.part1_out_chnls,
                                           kernel_size=1,
                                           stride=2)

        self.dim_equalizer1 = torch.nn.Conv2d(in_channels=self.part2_chnls,
                                              out_channels=self.part2_out_chnls,
                                              kernel_size=1)

        self.dim_equalizer2 = torch.nn.Conv2d(in_channels=self.part1_chnls,
                                              out_channels=self.part1_out_chnls,
                                              kernel_size=1)

        self.activation = activation()

    def forward(self, x):

        part1 = x[:, :self.part1_chnls, :, :] #part1 channel 
        part2 = x[:, self.part1_chnls:, :, :] #part2 channel 
        skip_connection = part2

        if self.stride == 2:
            skip_connection = self.projection1(skip_connection)
            part1 = self.projection2(part1)
        else:
            if self.part1_chnls != self.part1_out_chnls:
                skip_connection = self.dim_equalizer1(skip_connection)
                part1 = self.dim_equalizer2(part1)

        residual = self.residual_block(part2)  # F(x)
        residual = torch.add(residual, skip_connection)
        residual = self.activation(residual)
        out = torch.cat((part1, residual), 1)
        return out