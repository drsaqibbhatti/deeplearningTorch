import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.feedforward_block import feedforward_block
from block.attention_block import attention_block

class transformer_block(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                torch.nn.LayerNorm(dim),
                attention_block(dim, heads, dim_head, dropout),
                torch.nn.LayerNorm(dim),
                feedforward_block(dim, mlp_dim, dropout)
            ]))
        self.depth = depth

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            attn_norm = layer[0]
            attn = layer[1]
            ff_norm = layer[2]
            ff = layer[3]
            y = attn_norm(x)
            x = attn(y) + x
            y = ff_norm(x)
            x = ff(y) + x
        return x