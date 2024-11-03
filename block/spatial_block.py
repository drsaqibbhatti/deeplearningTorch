import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.channelpool_block import channelpool_block

class spatial_block(torch.nn.Module):
    def __init__(self):
        super(spatial_block, self).__init__()
        kernel_size = 7
        self.compress = channelpool_block()
        self.spatial = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size-1) // 2,
                            bias=False,
                            dilation=1),
            torch.nn.BatchNorm2d(num_features=1),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale