import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.transformer_block import transformer_block
from block.conv_bn_act_block import conv_bn_act_block

class mobilevit_block(torch.nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.activation = torch.nn.SiLU

        self.conv1 = conv_bn_act_block(inp=channel, oup=channel, kernal_size=kernel_size, activation=torch.nn.SiLU)
        self.conv2 = conv_bn_act_block(inp=channel, oup=dim, kernal_size=1, activation=torch.nn.SiLU)

        self.transformer = transformer_block(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_bn_act_block(inp=dim, oup=channel, kernal_size=1, activation=torch.nn.SiLU)
        self.conv4 = conv_bn_act_block(inp=2 * channel, oup=channel, kernal_size=kernel_size, activation=torch.nn.SiLU)


    def forward(self, x):
            y = x.clone()

            # Local representations
            x = self.conv1(x)
            x = self.conv2(x)

            # Global representations
            b, d, h, w = x.shape

            #     x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
            x = x.reshape(b, d, h//self.ph, self.ph, w//self.pw, self.pw).permute(0, 3, 5, 2, 4, 1).reshape(b,self.ph*self.pw, (h//self.ph)*(w//self.pw), d)

            x = self.transformer(x)

            #     x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
            x = x.reshape(b, self.ph, self.pw, h//self.ph, w//self.pw, d).permute(0, 5, 3, 1, 4, 2).reshape(b, d, h, w)

            # Fusion
            x = self.conv3(x)
            x = torch.cat((x, y), 1)
            x = self.conv4(x)

            return x