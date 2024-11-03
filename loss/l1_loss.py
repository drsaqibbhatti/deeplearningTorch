import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def l1_loss(prediction, label):
    mask = (label > 0).float()
    num_pos = torch.sum(mask) + torch.tensor(1e-4, dtype=torch.float32)
    loss = torch.abs(prediction - label) * mask
    loss = torch.sum(loss) / num_pos
    return loss