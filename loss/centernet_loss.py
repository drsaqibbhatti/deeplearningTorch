import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from loss.modified_focal_loss import modified_focal_loss
from loss.l1_loss import l1_loss

class centernet_loss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, beta=0.1):
        super(centernet_loss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.focal_loss = modified_focal_loss

    def forward(self, prediction_features,
                      prediction_sizemap,
                      prediction_offsetmap,
                      label_heatmap,
                      label_sizemap,
                      label_offsetmap):

        sum_class_loss = self.focal_loss(torch.sigmoid(prediction_features), label_heatmap) * self.alpha
        sum_size_loss = l1_loss(prediction_sizemap, label_sizemap)
        sum_offset_loss = l1_loss(prediction_offsetmap, label_offsetmap)

        return sum_class_loss + sum_size_loss + sum_offset_loss