import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.se_conv_block import se_conv_block
from block.spatial_block import spatial_block


class cbam_block(torch.nn.Module):
    def __init__(self, in_channels, channels, se_rate=0.5):
        super(cbam_block, self).__init__()
        #Squeeze-and-excitiation
        self.channel_wise_conv = se_conv_block(in_channels=in_channels,
                                               channels=channels,
                                               se_rate=se_rate)
        self.spatial_wise_conv = spatial_block()
    def forward(self, x):
        x_out = self.channel_wise_conv(x)
        x_out = self.spatial_wise_conv(x_out)
        return x_out