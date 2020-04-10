import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from livelossplot import PlotLosses

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def multi_gpu():
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        parallel_mode = True
        print('multi_GPU!!!!')
    else:
        parallel_mode = False
        print('single_GPU')

def train_loop(model, loss_fn, optimizer, num_epochs=10):
    
    model = model.to(device)
    multi_gpu()
    valid_loss = []
    
    for epoch in range(num_epochs):     
        
        scheduler.step()
        lr = scheduler.get_lr()[0]
        
        # train and valid loss
        data_train = train_valid(phase='train')
        data_valid = train_valid(phase='valid')
        
        # weight save
        valid_loss.append(data_valid[0])
        if np.array(valid_loss).min() == valid_loss[-1]:
            print('saving weight:',comment+'best.pt')
            if parallel_mode== False:
                torch.save({'state_dict': model.state_dict()}, comment+'best.pt')
            else:
                torch.save({'state_dict': model.module.state_dict()}, comment+'best.pt')
        print('epoch', epoch, 'lr', lr, data_valid[-1])
    
def train_seg(model, criterion, optimizer, num_epochs=100):
    liveloss = PlotLosses()
#     model = model.to(device)
#     multi_gpu()
    
    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_metric = 0.0

            for idx, batch in enumerate(dataloaders[phase]):
                inputs = batch['image'].to(device=device, dtype=torch.float)
                labels = batch['mask'].to(device=device, dtype=torch.float)
        
                preds = model(inputs)
                loss = criterion(preds, labels)
                running_loss += loss.item()
                
                metric = score_f1(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
                running_metric += metric.item()
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_metric = running_metric / len(dataloaders[phase])
            
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'loss'] = epoch_loss
            logs[prefix + 'metric'] = epoch_metric
        
        liveloss.update(logs)
        liveloss.draw()
        
# model = Recurrent(8)
# criterion = nn.CrossEntropyLoss()
# criterion = dice_loss
# train_segl(model, criterion, optimizer, num_epochs=100)
