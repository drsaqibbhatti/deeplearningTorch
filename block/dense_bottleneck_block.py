import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class dense_bottleneck_block(torch.nn.Module):
    def __init__(self, in_channels, growth_rate=32, expansion_rate=4, droprate=0.2, activation=torch.nn.ReLU):
        super().__init__()

        inner_channels = expansion_rate * growth_rate           ##expansion_size=32*4
        self.droprate = droprate

        self.residual = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),                  ##ex:128
            activation(),
            torch.nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),##32*4 #expansion layer
            torch.nn.BatchNorm2d(inner_channels),
            activation(),
            torch.nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False) ##32
        )

        self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        #output = F.dropout(self.residual(x), p=self.droprate, inplace=False, training=self.training)
        return torch.cat([self.shortcut(x), self.residual(x)], 1)