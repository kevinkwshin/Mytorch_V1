* To release memory, torch.cuda.empty_cache(). But it would be better to reset ur kernel.

https://github.com/BloodAxe/pytorch-toolbelt

# Define modules
```
# class DiceLoss(smp.utils.base.Loss):
#     def __init__(self,weight=None, **kwargs):
#         super().__init__(**kwargs)
#         self.weight = weight
        
#     def	forward(self, input, target):
#         assert input.shape == target.shape
#         N = target.size(0)
#         C = target.size(1)
#         smooth = 1
#         loss = 0.
#         if self.weight == None:
#             self.weight = [1] * C
            
#         input = input.view(N,C,-1)
#         target = target.view(N,C,-1)

#         for c in range(C):
#             iflat = input[:, c]
#             tflat = target[:, c]
#             intersection = (iflat * tflat).sum()

#             w = self.weight[c]
#             loss += w*(1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth)))
        
#         return loss
    
# class FocalTversky(smp.utils.base.Loss):
    
#     def __init__(self,alpha=0.7, gamma=0.75, weight=None, **kwargs):
#         super().__init__(**kwargs)
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight= weight
        
#     def forward(self, pr, gt):
#         eps = 1e-7
#         loss = 0 
#         N = gt.size(0)
#         C = gt.size(1)
#         pr = pr.view(N,C, -1)
#         gt = gt.view(N,C, -1) 
#         assert len(self.weight) == C
        
#         if self.weight == None:
#             self.weight = [1] * C
            
#         for c in range(C):
#             prc = pr[:, c]
#             gtc = gt[:, c]
        
#             true_pos = torch.sum(prc * gtc)
#             false_neg = torch.sum(gtc * (1-prc))
#             false_pos = torch.sum((1-gtc)*prc)
            
#             w = self.weight[c]
#             pt_1 = (true_pos + eps)/(true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + eps)
#             loss += torch.pow((1-pt_1), self.gamma) * w
        
#         return torch.mean(loss)

# class categorical_focal_loss(smp.utils.base.Loss):
#     r"""Implementation of Focal Loss from the paper in multiclass classification
#     Formula:
#         loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
#     Args:
#         gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         alpha: the same as weighting factor in balanced cross entropy, default 0.25
#         gamma: focusing parameter for modulating factor (1-p), default 2.0
#         class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
#     """

#     def __init__(self, alpha=0.75, gamma=4.0, **kwargs):
#         super().__init__(**kwargs)
#         self.alpha = alpha
#         self.gamma = gamma
    
#     def forward(self, pr, gt):
#         eps = 1e-7
#         pr = torch.clamp(pr, min=eps, max=1.0-eps)
#         loss = - gt * (self.alpha * torch.pow((1 - pr), self.gamma) * torch.log(pr))
        
#         return torch.mean(loss)
    

# loss1 = DiceLoss([.2,.2,.3,.3,.3])
loss1 = smp.utils.losses.FocalTversky(alpha=0.3, gamma=0.75, weight=[.25,.15,.25,.25,.1])
loss2 = smp.utils.losses.categorical_focal_loss(alpha=.25, gamma=4)

loss_seg = loss1 + loss2

loss = loss_seg

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.DICE(threshold=0.5),
    smp.utils.metrics.SENSITIVITY(threshold=0.5),
    smp.utils.metrics.SPECIFICITY(threshold=0.5),
]

initial_lr = 2e-4
num_epochs = 150

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=initial_lr),])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=device,
    verbose=True,
)

def model_save_state_dict(model,filename,parallel_mode):
    filename = filename+'.pt'
    print('saving weight:',filename)
    
    if parallel_mode== False:
        torch.save({'state_dict': model.state_dict()}, filename)
    else:
        torch.save({'state_dict': model.module.state_dict()}, filename)
```


```
from livelossplot import PlotLosses
plotlosses = PlotLosses()

score = []
for epoch in range(0, num_epochs):

    scheduler.step()
    lr = scheduler.get_lr()[0]
    indices = np.where(np.array(score) == np.array(score).min()) if len(score)>1 else [[0]]
    print('\nEpoch: [{:4}/{}]  lr : [{:.6f}]  Recently saved epoch : {} @ {}'.format(epoch,num_epochs, lr, indices[0], filename))

  train_logs = train_epoch.run(train_loader)
  valid_logs = valid_epoch.run(valid_loader)

#     score.append(valid_logs['loss_main']+valid_logs['loss_sub'])
#     if np.min(score) == valid_logs['loss_main']+valid_logs['loss_sub'] and epoch > 5:
#         model_save_state_dict(model,filename+str(epoch),parallel_mode)

  logkeys = list(train_logs)
  logs = {} 
  for logkey in logkeys:
      logs[logkey] = train_logs[logkey]; 
      logs['val_'+logkey] = valid_logs[logkey];#    del valid_logs[logkey];

  logkeys = list(logs)
  plotlosses.update({key_value : logs[key_value] for key_value in logkeys})
  plotlosses.send()
  
```
