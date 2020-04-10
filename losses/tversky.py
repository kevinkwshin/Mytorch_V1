import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import numpy as np

class BinaryTverskyLossV2(nn.Module):

    def __init__(self, alpha=.5,beta=.5, ignore_index=None, reduction='mean'):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative.
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.epsilon = 1e-8
        self.reduction = reduction
        s = self.beta + self.alpha
        if sum != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target):
        batch_size = output.size(0)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output).view(batch_size, -1)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * (1 - target), 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.epsilon)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss
