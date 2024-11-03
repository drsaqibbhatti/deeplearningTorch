import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple

class attention_block(torch.nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = torch.nn.Softmax(dim=-1)
        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

        self.hd = self.heads*dim_head

    def forward(self, x):
        b, p, n, d = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q = qkv[0].reshape(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        k = qkv[1].reshape(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        v = qkv[2].reshape(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, p, n, -1)

        return self.to_out(out)