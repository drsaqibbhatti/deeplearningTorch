import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class ws_conv_transpose(torch.nn.ConvTranspose2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 gain=True,
                 gamma=1.0,
                 eps=1e-5,
                 use_layernorm=False):

        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         output_padding=output_padding, groups=groups, bias=bias, dilation=dilation,
                         padding_mode=padding_mode)

        torch.nn.init.kaiming_normal_(self.weight)
        self.gain = torch.nn.Parameter(torch.ones(
            self.out_channels, 1, 1, 1)) if gain else None
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        # experimental, slightly faster/less GPU memory use
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None, eps: float = 1e-4) -> Tensor:
        return F.conv_transpose2d(input, self.get_weight(), self.bias, self.stride, self.padding, self.output_padding,
                                  self.groups, self.dilation)