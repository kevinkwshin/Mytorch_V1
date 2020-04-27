import torch, numpy as np
from torch import nn

class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name
class Metric(BaseObject):
    pass
    
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
    pr = _threshold(pr, threshold=threshold)
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
    
class DICE(Metric):
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
    
class IOU(Metric):
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

class SENSITIVITY(Metric):
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

class SPECIFICITY(Metric):
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
