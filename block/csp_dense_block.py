import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple

from block.dense_bottleneck_block import dense_bottleneck_block

class csp_dense_block(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, expansion_rate, growth_rate, droprate, part_ratio=0.5, activation=torch.nn.ReLU):
        super(csp_dense_block, self).__init__()


        self.part1_chnls = int(num_input_features * part_ratio)
        self.part2_chnls = num_input_features - self.part1_chnls ##Dense Layer Channel Calculation


        self.dense_block = torch.nn.Sequential()
        self.droprate = droprate

        for i in range(num_layers):
            layer = dense_bottleneck_block(in_channels=self.part2_chnls + i * growth_rate,
                                    growth_rate=growth_rate,
                                    expansion_rate=expansion_rate,
                                    droprate=droprate,
                                    activation=activation)
            self.dense_block.add_module('denselayer_%d' % (i + 1), layer)

        self.spatial_dropout = torch.nn.Dropout2d(p=self.droprate)

    def forward(self, x):

        part1 = x[:, :self.part1_chnls, :, :] #part1 channel 
        part2 = x[:, self.part1_chnls:, :, :] #part2 channel 
        part2 = self.dense_block(part2)
        #part2 = self.spatial_dropout(part2)
        out = torch.cat((part1, part2), 1)

        return out