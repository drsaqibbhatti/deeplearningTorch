import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class se_densebottleneck_block(torch.nn.Module):
    def __init__(self, in_channels, growth_rate=32, expansion_rate=4, droprate=0.2, activation=torch.nn.ReLU):
        super().__init__()

        inner_channels = expansion_rate * growth_rate  ##expansion_size=32*4
        self.droprate = droprate

        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),  ##32*4 #expansion layer
            torch.nn.BatchNorm2d(inner_channels),
            activation(),
            torch.nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False),  ##32
            torch.nn.BatchNorm2d(growth_rate),
            activation(),
        )

    def forward(self, x):
        return torch.cat([x, self.residual(x)], 1)