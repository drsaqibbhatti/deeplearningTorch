import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def modified_l1_loss(input: torch.Tensor, target: torch.Tensor, beta: float, weight_mask=None) -> torch.Tensor:

    mask = (target > 0).float()
    coordinate_num_points = torch.sum(mask)

    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target) * mask
    else:
        n = torch.abs(input - target) * mask
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)


    if weight_mask is not None:
        loss = loss * mask

    loss = loss.sum() / (coordinate_num_points + 1e-4)

    return loss