import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.dense_bottleneck_block import dense_bottleneck_block

class dense_block(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, expansion_rate, growth_rate, droprate, activation=torch.nn.ReLU):
        super(dense_block, self).__init__()

        self.dense_block = torch.nn.Sequential()
        self.droprate = droprate

        for i in range(num_layers):
            layer = dense_bottleneck_block(in_channels=num_input_features + i * growth_rate,
                                    growth_rate=growth_rate,
                                    expansion_rate=expansion_rate,
                                    droprate=droprate,
                                    activation=activation)
            self.dense_block.add_module('denselayer_%d' % (i + 1), layer)
        self.spatial_dropout = torch.nn.Dropout2d(p=self.droprate)


    def forward(self, x):
        x = self.dense_block(x)
        x = self.spatial_dropout(x)
        return x