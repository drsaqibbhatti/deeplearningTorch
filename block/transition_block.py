import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class transition_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.2, activation=torch.nn.ReLU):
        super().__init__()
        self.droprate=droprate
        self.down_sample = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            activation(),
            torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            torch.nn.AvgPool2d(2, stride=2)
        )
        self.spatial_dropout = torch.nn.Dropout2d(p=self.droprate)

    def forward(self, x):
        output = self.down_sample(x)
        return output