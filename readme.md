* To release memory, torch.cuda.empty_cache(). But it would be better to reset ur kernel.

https://github.com/BloodAxe/pytorch-toolbelt

# Define modules
```
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

# Train Loop

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
# Inference
```
model = smp.Unet(encoder_name='timm-efficientnet-b7',
                 decoder_attention_type='scse',
#                  aux_params=aux_params,
                 encoder_weights=None,
                 classes=5,
                 activation='softmax',)

model_w = torch.load(filename+'.pt')
model.load_state_dict(model_w['state_dict'])
model = model.to(device)

import ttach as tta
tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')```
