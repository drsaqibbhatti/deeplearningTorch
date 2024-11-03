import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class nc_transition_block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = torch.nn.Sequential(
            torch.nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        output = self.down_sample(x)
        return output