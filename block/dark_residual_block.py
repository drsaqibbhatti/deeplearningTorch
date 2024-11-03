import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class dark_residual_block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, activation=torch.nn.LeakyReLU):
        super(dark_residual_block, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            padding=0,
                            bias=False,
                            stride=stride),
            torch.nn.BatchNorm2d(num_features=out_channels),
            activation()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=in_channels),
            activation()
        )

    def forward(self, x):
        skip = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + skip
        return x