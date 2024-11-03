import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple

class mobilev2_block(torch.nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = torch.nn.Sequential(
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )
        else:
            self.conv = torch.nn.Sequential(
                # pw
                torch.nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(),
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)