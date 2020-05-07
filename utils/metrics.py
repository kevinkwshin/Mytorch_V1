from . import base
from . import functional as F
from .base import Activation
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score     

# def _threshold(x, threshold=None):
#     if threshold is not None:
#         return (x > threshold).type(x.dtype)
#     else:
#         return x

class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

    
    
def score_numeric(pr,gt,threshold):
    """Computation of statistical numerical scores:
    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives
    return: tuple (FP, FN, TP, TN)
    
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    pr = F._threshold(pr, threshold=threshold)
    batch_size = gt.size(0)
    
    TP = torch.sum(pr * gt)
    FP = torch.sum(pr * (1 - gt)) 
    FN = torch.sum((1 - pr) * gt)
    TN = torch.sum((1 - pr) * (1- gt))
    
    total = TP+FP+FN+TN
    TP /= total
    FP /= total
    FN /= total
    TN /= total
    
    return {'FP':FP,'FN':FN,'TP':TP, 'TN': TN}
    
class DICE(base.Metric):
    __name__ = 'dice_score'
    def __init__(self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.eps = eps
        
    def forward(self, y_pr, y_gt):
        scores = score_numeric(y_pr, y_gt,self.threshold)
        TP = scores['TP']
        FP = scores['FP']
        FN = scores['FN']
        TN = scores['TN']
        
        dice = 2*TP/(2*TP + FP + FN + self.eps)        
        return dice
    
class IOU(base.Metric):
    __name__ = 'iou_score'
    def __init__(self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.eps = eps
        
    def forward(self, y_pr, y_gt):
        scores = score_numeric(y_pr, y_gt,self.threshold)
        TP = scores['TP']
        FP = scores['FP']
        FN = scores['FN']
        TN = scores['TN']
        
        score =TP / (TP + FP + FN + self.eps)        
        return score

class SENSITIVITY(base.Metric):
    __name__ = 'sen_score'
    def __init__(self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.eps = eps
        
    def forward(self, y_pr, y_gt):
        scores = score_numeric(y_pr, y_gt,self.threshold)
        TP = scores['TP']
        FN = scores['FN']
        
        score = TP/ (TP + FN + self.eps)
        return score

class SPECIFICITY(base.Metric):
    __name__ = 'spe_score'
    def __init__(self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.eps = eps
        
    def forward(self, y_pr, y_gt):
        scores = score_numeric(y_pr, y_gt,self.threshold)
        TN = scores['TN']
        FP = scores['FP']
        
        score = TN/(TN + FP + self.eps)
        return score
    
class AUC(base.Metric):
    """
    input이 반드시 balance를 이루어여 함, 배치마다 반드시 모든 클래스 포함
    """
    __name__ = 'auc_score'
    def __init__(self, eps=1e-7, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        
    def forward(self, y_pr, y_gt):   
        
        y_gt = y_gt.cpu().detach().numpy()
        y_pr = y_pr.cpu().detach().numpy()
        score = roc_auc_score(y_gt,y_pr)
        return torch.tensor(score)