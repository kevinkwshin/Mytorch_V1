import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from livelossplot import PlotLosses
from tqdm import tqdm as tqdm

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def multi_gpu():
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        parallel_mode = True
        print('multi_GPU!!!!')
    else:
        parallel_mode = False
        print('single_GPU')

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for idx, batch in enumerate(iterator):
                
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                loss, y_pred = self.batch_update(x, y)
                
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss' : loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)                
                
                if idx % 10 == 0:
                    num_gridImage = 8

                    x = x[:,1]
                    x = x.unsqueeze(1)
#                     y_pred = y_pred[:,1]
#                     y_pred = y_pred.unsqueeze(1)

                    grid_x  = vutils.make_grid(x[:num_gridImage], nrow=4, normalize=True, scale_each=True)
                    grid_y  = vutils.make_grid(y[:num_gridImage], nrow=4, normalize=True, scale_each=True)
                    grid_y_pred  = vutils.make_grid(y_pred[:num_gridImage], nrow=4, normalize=True, scale_each=True)
                    writer.add_image(self.stage_name+'/x', grid_x, idx)
                    writer.add_image(self.stage_name+'/y', grid_y, idx)
                    writer.add_image(self.stage_name+'/y_pred', grid_y_pred, idx)      

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction       

class TestEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='test',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y,z):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
            prediction = torch.round(prediction)
            # do what you want
#             prediction = label_TPFPFN(prediction,y)  # 2nd label
#             np.save('Output/'+z[0].split('/')[-1],prediction.cpu().detach().numpy()[0]) 
            
        return loss, prediction

        
        
        

# optimizer = optim.Adam(model.parameters(), lr = initial_lr)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6, last_epoch=-1)
# loss = BinaryTverskyLossV2(.4,.6) + BinaryFocalLoss(gamma=6)
        
# model = Recurrent(8)
# criterion = nn.CrossEntropyLoss()
# criterion = dice_loss
# train_segl(model, criterion, optimizer, num_epochs=100)
