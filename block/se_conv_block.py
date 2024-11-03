import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class se_conv_block(torch.nn.Module):
    def __init__(self, in_channels, channels, se_rate=12):
        super(se_conv_block, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, int(channels // se_rate), kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(int(channels // se_rate)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int(channels // se_rate), channels, kernel_size=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y