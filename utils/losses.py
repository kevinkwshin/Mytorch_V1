import torch
import torch.nn as nn

from . import base
from . import functional as F
from  .base import Activation

class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

# class BinaryFocalLoss(base.Loss):
#     def __init__(self, alpha=0.75, gamma=4.0, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.alpha = alpha
#         self.gamma = gamma
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.binary_focal_loss(
#             y_pr, y_gt,
#             gamma=self.gamma,
#             alpha=self.alpha,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )

EPSILON = 1e-6

class StableBCELoss(base.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
#         super(StableBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
    
class BinaryFocalLoss(base.Loss):
    def __init__(self, alpha=[0.75, 0.25], gamma=4.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.binary_focal_loss(
            y_pr, y_gt,
            gamma=self.gamma,
            alpha=self.alpha,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
    
    
class TverskyLoss(base.Loss):
    def __init__(self, beta=0.8, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.tversky_loss(
            y_pr, y_gt,
            beta=self.beta,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

    
    
class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
