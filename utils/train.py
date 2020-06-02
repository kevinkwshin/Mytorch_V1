import sys
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
from .metrics import IOU

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, iteration=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.iteration = iteration
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
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
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
            metrics_meters_aux = {metric.__name__+'_aux': AverageValueMeter() for metric in self.metrics}
#             metrics_meters = {**metrics_meters, **metrics_meters_aux}
        else:
#             loss_meter = AverageValueMeter()
            loss_meter = {'loss_seg':AverageValueMeter()}
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for iteration,batch in enumerate(iterator):
                if self.iteration is not None:
                    if self.iteration < iteration:
                        break
                        
                x = batch['data'].to(self.device)
                y = batch['seg'].to(self.device)
                z = batch['cls'].to(self.device)
                
                if isinstance(self.loss, list):                
                    loss, loss_aux, y_pred, y_pred_aux = self.batch_update(x, y, z)
                    loss_value, loss_aux_value = loss.cpu().detach().numpy(), loss_aux.cpu().detach().numpy()
                    loss_meter['loss_seg'].add(loss_value)
                    loss_meter['loss_cls'].add(loss_aux_value)
                    loss_logs = {'loss_seg': loss_meter['loss_seg'].mean,'loss_cls': loss_meter['loss_cls'].mean}
                    
                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                        metric_value_aux = metric_fn(y_pred_aux, z).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                        metrics_meters_aux[metric_fn.__name__+'_aux'].add(metric_value_aux)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    metrics_logs_aux = {k: v.mean for k, v in metrics_meters_aux.items()}
                    metrics_logs = {**metrics_logs, **metrics_logs_aux}
                    
                else:
                    loss, y_pred = self.batch_update(x, y, z)
                    loss_value = loss.cpu().detach().numpy()
#                     loss_meter.add(loss_value)
#                     loss_logs = {self.loss.__name__: loss_meter.mean}
                    loss_meter['loss_seg'].add(loss_value)
                    loss_logs = {'loss_seg': loss_meter['loss_seg'].mean}
        
                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    
                logs.update(loss_logs)

#                 # update metrics logs
#                 for metric_fn in self.metrics:
#                     metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
#                     metric_value_aux = metric_fn(y_pred_aux, y).cpu().detach().numpy()
#                     metrics_meters[metric_fn.__name__].add(metric_value)
#                     metrics_meters_aux[metric_fn.__name__].add(metric_value_aux)
#                 metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, iteration =None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            iteration = iteration
        )
        self.optimizer = optimizer
        
    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, z):
        self.optimizer.zero_grad()
        
        if isinstance(self.loss, list):
            prediction,prediction_aux = self.model(x)
            loss_main_f, loss_sub_f = self.loss 
            loss_main = loss_main_f(prediction, y)
            loss_sub = loss_sub_f(prediction_aux, z)
            loss = loss_main + loss_sub
            loss.backward()
            self.optimizer.step()
            return loss_main, loss_sub, prediction, prediction_aux
        else:
            prediction = self.model(x)
            loss = self.loss(prediction, y)
            loss.backward()
            self.optimizer.step()
            return loss, prediction

class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True, iteration =None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            iteration = iteration
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, z):
        with torch.no_grad():
    
            if isinstance(self.loss, list):
                prediction,prediction_aux = self.model(x)
                loss_main_f, loss_sub_f = self.loss 
                loss_main = loss_main_f(prediction, y)
                loss_sub = loss_sub_f(prediction_aux, z)
                return loss_main, loss_sub, prediction, prediction_aux
            else:
                prediction = self.model(x)
                loss = self.loss(prediction, y)
                return loss, prediction