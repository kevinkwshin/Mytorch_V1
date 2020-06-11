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

# class DiceLoss(base.Loss):

#     def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.f_score(
#             y_pr, y_gt,
#             beta=self.beta,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )

class BinaryFocalLoss(base.Loss):
    def __init__(self, alpha=0.75, gamma=4.0, activation=None, ignore_channels=None, **kwargs):
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

class StableBCELoss(base.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
    
    
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
    
class DiceLoss(base.Loss):
    def __init__(self,weight=None):
        super(DiceLoss, self).__init__()
        self.weight = weight
        
    def	forward(self, input, target):
        assert input.shape == target.shape
        N = target.size(0)
        C = target.size(1)
        smooth = 1
        loss = 0.
        if self.weight == None:
            self.weight = torch.ones(C)
        
        input = input.view(N,C,-1)
        target = target.view(N,C,-1)
        
        for c in range(C):
            
            iflat = input[:, c].view(-1)
            tflat = target[:, c].view(-1)
            intersection = (iflat * tflat).sum()

            w = self.weight[c]
            loss += w*(1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth)))
        
        return loss
    
class BinaryFocalLoss(base.Loss):
    
    def __init__(self, alpha=0.75, gamma=4.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pr, gt):
        eps = 1e-7
        
        pr = torch.clamp(pr, min=eps, max=1.0-eps)
        loss_1 = - gt * (self.alpha * torch.pow((1 - pr), self.gamma) * torch.log(pr))
        loss_0 = - (1 - gt) * ((1 - self.alpha) * torch.pow((pr), self.gamma) * torch.log(1 - pr))   # 여기서는 alpha에 가중치를 줘서 FN을 떨어뜨림!!
        loss = loss_0 + loss_1
        
        return torch.mean(loss)
    
class TverskyLoss(base.Loss):
    def __init__(self, beta = 0.5):
        super(TverskyLoss,self).__init__()
        self.beta = beta
        
    def forward(self, input, target):
        N = target.size(0)
        eps = 1
        
        pr = input.view(N, -1)
        gt = target.view(N, -1)
        
        tp = torch.sum(pr * gt,1)
        fp = torch.sum(pr * (1-gt),1)
        fn = torch.sum((1-pr) * gt,1)
        
        loss = 1 - (tp + eps) / (tp + (1-self.beta)*fp + self.beta*fn + eps)   # FN에 가중치를 줘서 FN을 떨어뜨림!
        
        return torch.mean(loss)

class categorical_focal_loss(base.Loss):
    r"""Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    """

    def __init__(self, alpha=0.75, gamma=4.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pr, gt):
        eps = 1e-7
                
        pr = torch.clamp(pr, min=eps, max=1.0-eps)
        loss = - gt * (self.alpha * torch.pow((1 - pr), self.gamma) * torch.log(pr))
        
        return torch.mean(loss)
    
    
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
