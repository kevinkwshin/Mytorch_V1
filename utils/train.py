import sys
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
from .metrics import IOU

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
#         self.loss.to(self.device)
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
            
        if isinstance(self.loss, list):
            loss_meter = {'loss_seg':AverageValueMeter(),'loss_cls':AverageValueMeter()}
        else:
            loss_meter = AverageValueMeter()
        
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, z, filename in iterator:
                x, y, z, filename = x.to(self.device), y.to(self.device), z.to(self.device), filename
                    
                if isinstance(self.loss, list):                
                    loss, loss_aux, y_pred, y_pred_aux = self.batch_update(x, y, z)
                    loss_value, loss_aux_value = loss.cpu().detach().numpy(), loss_aux.cpu().detach().numpy()
                    loss_meter['loss_seg'].add(loss_value)
                    loss_meter['loss_cls'].add(loss_aux_value)
                    loss_logs = {'loss_seg': loss_meter['loss_seg'].mean,'loss_cls': loss_meter['loss_cls'].mean}
                else:
                    loss, y_pred = self.batch_update(x, y, z)
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {self.loss.__name__: loss_meter.mean}

                # Orignal code
#                 # update loss logs
#                 loss_value = loss.cpu().detach().numpy()
#                 loss_meter.add(loss_value)
#                 loss_logs = {self.loss.__name__: loss_meter.mean}
                    
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    if metric_fn.__name__ =='auc_score':
                        metric_value = metric_fn(y_pred_aux, z).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    else:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

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

    def batch_update(self, x, y, z):
        self.optimizer.zero_grad()
        
        if isinstance(self.loss, list):
#             prediction,prediction_aux = self.model.forward(x)
            prediction,prediction_aux = self.model(x)
            loss_main_f, loss_sub_f = self.loss 
            loss_main = loss_main_f(prediction, y)
            loss_sub = loss_sub_f(prediction_aux, z)
            loss = loss_main + loss_sub
            loss.backward()
            self.optimizer.step()
            return loss_main, loss_sub, prediction, prediction_aux
        else:
#             prediction = self.model.forward(x)
            prediction = self.model(x)
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

    def batch_update(self, x, y, z):
        with torch.no_grad():
    
            if isinstance(self.loss, list):
#                 prediction,prediction_aux = self.model.forward(x)
                prediction,prediction_aux = self.model(x)
                loss_main_f, loss_sub_f = self.loss 
                loss_main = loss_main_f(prediction, y)
                loss_sub = loss_sub_f(prediction_aux, z)
                return loss_main, loss_sub, prediction, prediction_aux
            else:
#                 prediction = self.model.forward(x)
                prediction = self.model(x)
                loss = self.loss(prediction, y)
                return loss, prediction