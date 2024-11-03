import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from loss.modified_focal_loss import modified_focal_loss
from loss.modified_l1_loss import modified_l1_loss

class centernetv2_loss(torch.nn.Module):
    def __init__(self, lambda_offset=1.0, lambda_size=0.1, beta=1.0):
        super(centernetv2_loss, self).__init__()

        self.lambda_offset = lambda_offset
        self.lambda_size = lambda_size
        self.beta = beta
        self.focal_loss = modified_focal_loss

    def forward(self, prediction_features,
                      prediction_sizemap,
                      prediction_offsetmap,
                      label_heatmap,
                      label_sizemap,
                      label_offsetmap,
                      label_weight_mask):

        sum_class_loss = self.focal_loss(torch.sigmoid(prediction_features), label_heatmap)
        sum_size_loss = modified_l1_loss(prediction_sizemap, label_sizemap, beta=self.beta,
                                                weight_mask=label_weight_mask)
        sum_offset_loss = modified_l1_loss(prediction_offsetmap, label_offsetmap, beta=self.beta,
                                                  weight_mask=label_weight_mask)

        return sum_class_loss + (sum_size_loss * self.lambda_size) + (sum_offset_loss * self.lambda_offset)