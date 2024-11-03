import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class channelpool_block(torch.nn.Module):
    def forward(self, x):
        #Channel Wise Max pooling and Average pooling concat
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)