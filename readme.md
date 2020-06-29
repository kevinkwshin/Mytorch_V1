torch.cuda.empty_cache()

https://github.com/BloodAxe/pytorch-toolbelt
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
