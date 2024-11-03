import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple

def conv_bn_act_block(inp, oup, kernal_size=3, stride=1, activation=torch.nn.SiLU):
    if stride==1 and kernal_size==1:
        return torch.nn.Sequential(
            torch.nn.Conv2d(inp, oup, kernal_size, stride, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
            activation()
        )
    else :
        return torch.nn.Sequential(
            torch.nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
            torch.nn.BatchNorm2d(oup),
            activation()
        )
