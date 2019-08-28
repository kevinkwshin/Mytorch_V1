import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from livelossplot import PlotLosses

def train_model(model, criterion, optimizer, num_epochs=100):
    liveloss = PlotLosses()
    model = model.to(device)
    
    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_metric = 0.0

            for idx, batch in enumerate(dataloaders[phase]):
                inputs = batch['image'].to(device=device, dtype=torch.float)
                labels = batch['mask'].to(device=device, dtype=torch.float)
        
                preds = model(inputs)
                loss = criterion(preds, labels)
                running_loss += loss.item()
                metric = f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
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
# train_model(model, criterion, optimizer, num_epochs=100)
