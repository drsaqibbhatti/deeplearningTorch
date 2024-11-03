import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple



class se_block(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, activation=torch.nn.ReLU):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels//reduction_ratio),
            activation(),
            torch.nn.Linear(in_channels//reduction_ratio, in_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x