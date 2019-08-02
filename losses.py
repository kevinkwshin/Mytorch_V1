import torch
import torch.nn as nn
from torch.nn import Module

def dice_loss(input, target):
    """Dice loss.
    :param input: The input (predicted)
    :param target:  The target (ground truth)
    :returns: the Dice score between 0 and 1.
    """
    eps = 0.0001

    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)
    return - dice

# def dice_loss(true, logits, eps=1e-7):
#     """Computes the Sørensen–Dice loss.
#     Note that PyTorch optimizers minimize a loss. In this
#     case, we would like to maximize the dice loss so we
#     return the negated dice loss.
#     Args:
#         true: a tensor of shape [B, 1, H, W].
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         eps: added to the denominator for numerical stability.
#     Returns:
#         dice_loss: the Sørensen–Dice loss.
#     """
#     num_classes = logits.shape[1]
#     if num_classes == 1:
#         true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(logits, dim=1)
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0,) + tuple(range(2, true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     cardinality = torch.sum(probas + true_1_hot, dims)
#     dice_loss = (2. * intersection / (cardinality + eps)).mean()
#     return (1 - dice_loss)

class MaskedDiceLoss(Module):
    """A masked version of the Dice loss.
    :param ignore_value: the value to ignore.
    """

    def __init__(self, ignore_value=-100.0):
        super().__init__()
        self.ignore_value = ignore_value

    def forward(self, input, target):
        eps = 0.0001

        masking = target == self.ignore_value
        masking = masking.sum(3).sum(2)
        masking = masking == 0
        masking = masking.squeeze()

        labeled_target = target.index_select(0, masking.nonzero().squeeze())
        labeled_input = input.index_select(0, masking.nonzero().squeeze())

        iflat = labeled_input.view(-1)
        tflat = labeled_target.view(-1)

        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum()

        dice = (2.0 * intersection + eps) / (union + eps)

        return - dice
    
class ConfidentMSELoss(Module):
    def __init__(self, threshold=0.96):
        self.threshold = threshold
        super().__init__()

    def forward(self, input, target):
        n = input.size(0)
        conf_mask = torch.gt(target, self.threshold).float()
        input_flat = input.view(n, -1)
        target_flat = target.view(n, -1)
        conf_mask_flat = conf_mask.view(n, -1)
        diff = (input_flat - target_flat)**2
        diff_conf = diff * conf_mask_flat
        loss = diff_conf.mean()
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
# class FocalLoss(nn.Module):
#     def __init__(self, num_classes=20):
#         super(FocalLoss, self).__init__()
#         self.num_classes = num_classes

#     def focal_loss(self, x, y):
#         '''Focal loss.
#         Args:
#           x: (tensor) sized [N,D].
#           y: (tensor) sized [N,].
#         Return:
#           (tensor) focal loss.
#         '''
#         alpha = 0.25
#         gamma = 2

#         t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
#         t = t[:,1:]  # exclude background
#         t = Variable(t).cuda()  # [N,20]

#         p = x.sigmoid()
#         pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
#         w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
#         w = w * (1-pt).pow(gamma)
#         return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

#     def focal_loss_alt(self, x, y):
#         '''Focal loss alternative.
#         Args:
#           x: (tensor) sized [N,D].
#           y: (tensor) sized [N,].
#         Return:
#           (tensor) focal loss.
#         '''
#         alpha = 0.25

#         t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
#         t = t[:,1:]
#         t = Variable(t).cuda()

#         xt = x*(2*t-1)  # xt = x if t > 0 else -x
#         pt = (2*xt+1).sigmoid()

#         w = alpha*t + (1-alpha)*(1-t)
#         loss = -w*pt.log() / 2
#         return loss.sum()

#     def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
#         '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
#         Args:
#           loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
#           loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
#           cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
#           cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
#         loss:
#           (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
#         '''
#         batch_size, num_boxes = cls_targets.size()
#         pos = cls_targets > 0  # [N,#anchors]
#         num_pos = pos.data.long().sum()

#         ################################################################
#         # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
#         ################################################################
#         mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
#         masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
#         masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
#         loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

#         ################################################################
#         # cls_loss = FocalLoss(loc_preds, loc_targets)
#         ################################################################
#         pos_neg = cls_targets > -1  # exclude ignored anchors
#         mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
#         masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
#         cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

#         print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
#         loss = (loc_loss+cls_loss)/num_pos
#         return loss
