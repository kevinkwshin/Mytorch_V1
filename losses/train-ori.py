import sys
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
from .metrics import IOU
from .losses import BinaryFocalLoss

class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
    
class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, auxilary=False):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.auxilary = auxilary
        
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
        if self.auxilary == False:
            loss_meter = AverageValueMeter()
        elif self.auxilary == True:
            loss_meter = {'loss_main':AverageValueMeter(),'loss_sub':AverageValueMeter()}
            
#         if isinstance(loss, list):
#             loss_meter = {'loss_main':AverageValueMeter(),'loss_sub':AverageValueMeter()}
#         else:
#             loss_meter = AverageValueMeter()
        
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, z in iterator:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                if self.auxilary == False:
                    loss, y_pred = self.batch_update(x, y, z)
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {self.loss.__name__: loss_meter.mean}
                elif self.auxilary == True:
                    loss, loss_aux, y_pred, y_pred_aux = self.batch_update(x, y, z)
                    loss_value, loss_aux_value = loss.cpu().detach().numpy(), loss_aux.cpu().detach().numpy()
                    loss_meter['loss_main'].add(loss_value)
                    loss_meter['loss_sub'].add(loss_aux_value)
                    loss_logs = {'loss_main': loss_meter['loss_main'].mean,'loss_sub': loss_meter['loss_sub'].mean}
                    
#                 if isinstance(loss, list):                
#                     loss, loss_aux, y_pred, y_pred_aux = self.batch_update(x, y, z)
#                     loss_value, loss_aux_value = loss.cpu().detach().numpy(), loss_aux.cpu().detach().numpy()
#                     loss_meter['loss_main'].add(loss_value)
#                     loss_meter['loss_sub'].add(loss_aux_value)
#                     loss_logs = {'loss_main': loss_meter['loss_main'].mean,'loss_sub': loss_meter['loss_sub'].mean}
#                 else:
#                     loss, y_pred = self.batch_update(x, y, z)
#                     loss_value = loss.cpu().detach().numpy()
#                     loss_meter.add(loss_value)
#                     loss_logs = {self.loss.__name__: loss_meter.mean}

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

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, auxilary=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.auxilary = auxilary
        
    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, z):
        self.optimizer.zero_grad()
        
        if self.auxilary==False:
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
            loss.backward()
            self.optimizer.step()
            return loss, prediction
        elif self.auxilary==True:
            prediction,prediction_aux = self.model.forward(x)
            loss_main = self.loss(prediction, y)
#             loss_sub = StableBCELoss()(prediction_aux, z)
            loss_sub = BinaryFocalLoss(alpha=[0.5, 0.5], gamma=4.0)(prediction_aux, z)+ StableBCELoss()(prediction_aux, z)
            loss = loss_main + loss_sub
            loss.backward()
            self.optimizer.step()
            return loss_main, loss_sub, prediction, prediction_aux

        
#         if isinstance(self.loss, list):
#             prediction,prediction_aux = self.model.forward(x)
#             loss_main = self.loss(prediction, y)
# #             loss_sub = StableBCELoss()(prediction_aux, z)
#             loss_sub = BinaryFocalLoss(alpha=[0.5, 0.5], gamma=4.0)(prediction_aux, z)
#             loss = loss_main + loss_sub
#             loss.backward()
#             self.optimizer.step()
#             return loss_main, loss_sub, prediction, prediction_aux
#         else:
#             prediction = self.model.forward(x)
#             loss = self.loss(prediction, y)
#             loss.backward()
#             self.optimizer.step()
#             return loss, prediction

class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True, auxilary=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.auxilary = auxilary

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, z):
        with torch.no_grad():
            
            
            if self.auxilary==False:
                prediction = self.model.forward(x)
                loss = self.loss(prediction, y)
                return loss, prediction
            elif self.auxilary==True:
                prediction,prediction_aux = self.model.forward(x)
                loss_main = self.loss(prediction, y)
#                 loss_sub = StableBCELoss()(prediction_aux, z)
                loss_sub = BinaryFocalLoss(alpha=[0.5, 0.5], gamma=4.0)(prediction_aux, z)+ StableBCELoss()(prediction_aux, z)
                return loss_main, loss_sub, prediction, prediction_aux
    
#             if isinstance(self.loss, list):
#                 prediction,prediction_aux = self.model.forward(x)
#                 loss_main = self.loss(prediction, y)
# #                 loss_sub = StableBCELoss()(prediction_aux, z)
#                 loss_sub = BinaryFocalLoss(alpha=[0.5, 0.5], gamma=4.0)(prediction_aux, z)
#                 return loss_main, loss_sub, prediction, prediction_aux
#             else:
#                 prediction = self.model.forward(x)
#                 loss = self.loss(prediction, y)
#                 return loss, prediction