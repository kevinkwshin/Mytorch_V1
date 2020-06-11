import torch

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


# def binary_focal_loss(pr, gt, gamma=4.0, alpha=0.75, eps=1e-7, threshold=None, ignore_channels=None):
#     r"""Implementation of Focal Loss from the paper in binary classification
#     Formula:
#         loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
#                - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)
#     Args:
#         gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         alpha: the same as weighting factor in balanced cross entropy, default 0.25
#         gamma: focusing parameter for modulating factor (1-p), default 2.0
#     """
#     pr = _threshold(pr, threshold=threshold)
#     pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    
#     pr = torch.clamp(pr, min=eps, max=1.0-eps)
#     loss_1 = - gt * (alpha * torch.pow((1 - pr), gamma) * torch.log(pr))
#     loss_0 = - (1 - gt) * ((1 - alpha) * torch.pow((pr), gamma) * torch.log(1 - pr))   # 여기서는 alpha에 가중치를 줘서 FN을 떨어뜨림!!
#     loss = loss_0 + loss_1
#     result_loss = torch.sum(loss, dim=(0,2,3))
    
#     return torch.mean(result_loss)


# def binary_focal_loss(pr, gt, gamma=2.0, alpha=[0.5,0.5], eps=1e-7, threshold=None, ignore_channels=None):
#     r"""Implementation of Focal Loss from the paper in binary classification
#     Formula:
#         loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
#                - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)
#     Args:
#         gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         alpha: the same as weighting factor in balanced cross entropy, default 0.25
#         gamma: focusing parameter for modulating factor (1-p), default 2.0
#     """
#     pr = _threshold(pr, threshold=threshold)
#     pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    
#     pr = torch.clamp(pr, min=eps, max=1.0-eps)
    
#     pos_mask = (gt == 1).float()
#     neg_mask = (gt == 0).float()
        
#     pos_loss = -alpha[0] * torch.pow(torch.sub(1.0, pr), gamma) * torch.log(pr) * pos_mask
#     neg_loss = -alpha[1] * torch.pow(pr, gamma) * torch.log(torch.sub(1.0, pr)) * neg_mask

#     neg_loss = neg_loss.sum()
#     pos_loss = pos_loss.sum()
#     num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
#     num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

#     if num_pos == 0:
#         loss = neg_loss
#     else:
#         loss = pos_loss / num_pos + neg_loss / num_neg
#     return loss

def binary_focal_loss(pr, gt, gamma=4.0, alpha=0.7, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    
#     print("focal pr.shape = ", pr.shape) # torch.Size([8, 1, 320, 320])
#     print("focal gt.shape = ", gt.shape) # torch.Size([8, 1, 320, 320])

    # clip to prevent NaN's and Inf's
    pr = torch.clamp(pr, min=eps, max=1.0-eps)
    
    loss_1 = - gt * (alpha * torch.pow((1 - pr), gamma) * torch.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * torch.pow((pr), gamma) * torch.log(1 - pr))   # 여기서는 alpha에 가중치를 줘서 FN을 떨어뜨림!!
    loss = loss_0 + loss_1
#     print("focal_loss.shape = ", loss.shape)  # torch.Size([8, 1, 320, 320])
#     print("torch.mean(loss).shape", torch.mean(loss).shape)  # torch.Size([])
#     result_loss = torch.sum(loss, dim=(0,2,3))
    return torch.mean(loss)

def tversky_loss(pr, gt, beta=0.8, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    tp = torch.sum(gt * pr, dim=(0,2,3))
    fp = torch.sum(pr, dim=(0,2,3)) - tp
    fn = torch.sum(gt, dim=(0,2,3)) - tp
    loss = 1 - (2*tp + eps) / (2*tp + (1-beta)*fp + beta*fn + eps)   # FN에 가중치를 줘서 FN을 떨어뜨림!
    
    return torch.mean(loss)
