* To release memory, torch.cuda.empty_cache(). But it would be better to reset ur kernel.

https://github.com/BloodAxe/pytorch-toolbelt

# GPU Setting
```
!nvidia-smi
gpus= "7"

import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]= gpus;

import torch
gpu_count = torch.cuda.device_count()
print(torch.cuda.is_available())
print(gpu_count)
```

# Import Library
```
!pip install -r Mytorch/requirements.txt
import Mytorch
```

# Parameter Setting & Research Note
```
!sudo pip install pytorch_lightning torch_optimizer wandb --quiet

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
import torch_optimizer
import torchvision
import wandb
import torch.nn as nn

class compoundloss(nn.Module):    
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, pr, gt):
        loss3= Mytorch.utils.losses.BinaryFocalLoss(alpha=.5, gamma=2) #FocalTversky(alpha=0.4, gamma=0.75, weight=[.1])
        loss = loss3(pr,gt)
    
        return torch.mean(loss)

checkpoint_callback = ModelCheckpoint(
#     filepath='test.ckpt',
    verbose=True,
    monitor='val_loss',
    mode='min'
)

seed_everything(1)
loss = nn.BCEWithLogitsLoss()

distributed_backend = 'dp' if gpu_count>1 else None
# sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label=for_sampler_get_label)
sampler = None

resume_from_checkpoint = None
# resume_from_checkpoint = '/workspace/Brain_Hemo/Code/brain_hemo/EfficientNet_BCEWithLogitsLoss/checkpoints/epoch=12.ckpt'
metric = pl.metrics.Accuracy()

config = {
    "project_name":"Tendon_seg",
    "exam_name":net.__repr__().split('(')[0]+'_'+loss.__repr__()[:-2],    
    "resume_from_checkpoint":resume_from_checkpoint,
    "learning_rate": 5e-4,
    "epoch": 1000,
    "batch_size": 32,
    "amp_level":'O1',
    "precision":16,
    "metric": metric,
    "loss": loss,
    "gpus":gpus,
    "net":net,    
    "distributed_backend":distributed_backend,
#     "auto_scale_batch_size":False,
    "auto_scale_batch_size":True,
    "sampler":sampler,
}


def experiment_name(config):
    exam_name = ''+'_'+loss.__repr__()[:-2]+'_'
    return exam_name

exam_name = experiment_name(config)

logger = WandbLogger(project=config['project_name'],
                     name=config['exam_name'],
                     id=config['exam_name'],
                    )
```


# Dataloader
```

class Dataset_npy():#DataLoader):
    def __init__(self, x_list, y_list, augmentation=None):
        self.x_list = x_list
        self.y_list = y_list
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.x_list)
  
    def __getitem__(self, index):
        
        image = np.load(self.x_list[index])
        image = image_preprocess_float(image)
        gt_seg = np.load(self.y_list[index])
        gt_seg = gt_seg[...,1]
        
        gt_seg[gt_seg==2] = 1
        gt_seg[gt_seg==5] = 4
        class_values = [1,3,4,6]
#         class_values = [1,2,3,4,5,6]
        gt_segs = [(gt_seg == v) for v in class_values]
        gt_seg = np.stack(gt_segs, axis=-1).astype('float')
        
        # add background if mask is not binary
        if gt_seg.shape[-1] != 1:
            background = 1 - gt_seg.sum(axis=-1, keepdims=True)
            gt_seg = np.concatenate((background,gt_seg), axis=-1)
    
        if self.augmentation:
            sample = self.augmentation(image = image, mask = gt_seg)
            image, gt_seg = sample['image'], sample['mask']
            
            n = np.random.randint(0,2,1)[0]
            if n ==1:
                image_ = image.copy()
                image_[:,:,0] = image[:,:,2]
                image_[:,:,2] = image[:,:,0]
                image = image_
                
        if np.any(gt_seg):
            gt = torch.tensor([1.],dtype=float)
        else:
            gt = torch.tensor([0.],dtype=float)

        image = np.moveaxis(image,-1,0)    
        gt_seg = np.moveaxis(gt_seg,-1,0)
        
        return {"data":image, "seg":gt_seg, "cls":gt, "fname":self.x_list[index]}
                
```
class SegModel_plain(pl.LightningModule):

    def __init__(self,config, **kwargs):
        super().__init__()
        
        self.net = config['net']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.loss = config['loss']
        self.metric = config['metric']
        
        self.save_hyperparameters()        

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        
        x = batch['data']
        y = batch['seg']
        pred = self(x)
        loss = self.loss(pred,y)
        
        logs = {"train_loss": loss,}# "train_accuracy": accuracy,"train_dice":dice}#, "train_auc":auc}

        return {'loss': loss,
#                 'log': log_dict,
                'progress_bar': logs,
#                 'x': x, 'y': y, 'pred': pred
               }
    
    def validation_step(self, batch, batch_idx):
        x = batch['data']
        y = batch['seg']
        pred = self(x)
        loss = self.loss(pred,y)
        
#         loss = F.binary_cross_entropy_with_logits(pred_cls,y_cls)
        pred = F.sigmoid(pred)
    
        return {'val_loss': loss,
                'x':x,
                'y':y,
                'pred':pred
               }
    
    def batch_prediction(self, inputs, batch_size):

        y_preds=[]
        total_batch  = int(np.ceil(len(inputs)/batch_size))
        for idx in range(total_batch):
            batch_ = inputs[idx*batch_size:(idx+1)*batch_size]
            y_pred = self.forward(batch_)
            y_preds.append(y_pred)

        y_preds = torch.cat(y_preds,dim=0)
        return y_preds
    
    def test_step(self, batch, batch_idx):
        
        x = batch['data'].squeeze(0)
        y_seg = batch['seg'].squeeze(0)
        pred_cls = self(x)
        loss = self.loss(pred_cls,y_cls)
#         loss = F.binary_cross_entropy_with_logits(pred_cls,y_cls)
        pred_cls = F.sigmoid(pred_cls)
    
        return {'loss_test': loss, 
                'x':x,
                'y':y,
                'pred':pred
               }

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        x = torch.cat([x['x'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs]).cpu().squeeze()
        pred = torch.cat([x['pred'] for x in outputs]).cpu().squeeze()
                
#         accuracy = pl.metrics.functional.accuracy(pred_cls.round(),y_cls)
#         dice = pl.metrics.functional.dice_score(pred_cls.round(),y_cls)
# #         auc = pl.metrics.functional.auroc(pred_cls.round(),y_cls)
    
#         logs = {"val_loss": val_loss_mean, "valid_accuracy": accuracy,"valid_dice":dice}#, "valid_auc":auc}

        try:
            acc = sklearn.metrics.accuracy_score(y_cls,pred_cls.round())
            f1 = sklearn.metrics.f1_score(y_cls,pred_cls.round())
            recall=sklearn.metrics.recall_score(y_cls,pred_cls.round())
            AUROC = sklearn.metrics.roc_auc_score(y_cls,pred_cls)
            cm = sklearn.metrics.confusion_matrix(y_cls,pred_cls.round())

            print("epoch {} acc {} recall {} f1 {} AUROC {} confusion matrix:\n {}".format(self.current_epoch,acc,recall,f1,AUROC,cm))
        except:
            pass
        
        logs = {"val_loss": val_loss_mean,}
        return {'log':logs}
    
    
    def test_epoch_end(self, outputs):
#         test_loss_mean = torch.stack([x['loss_test'] for x in outputs]).mean()
                    
        x = torch.cat([x['x'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs]).cpu().squeeze().numpy()
        pred = torch.cat([x['pred'] for x in outputs]).cpu().squeeze().numpy()
        
        print('acc:',sklearn.metrics.accuracy_score(y_cls,pred_cls.round()))
        print('f1:',sklearn.metrics.f1_score(y_cls.squeeze(),pred_cls.round()))
        print('AUC:',sklearn.metrics.roc_auc_score(y_cls.squeeze(),pred_cls))   
        print('confusion matrix:\n',sklearn.metrics.confusion_matrix(y_cls,pred_cls.round()))
        
        cm = sklearn.metrics.confusion_matrix(y_cls,pred_cls.round())
        sklearn.metrics.ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        
        print(sklearn.metrics.classification_report(y_cls,pred_cls.round()))

        return {
#                 'loss_test':test_loss_mean,
#                 'outputs':outputs,
               }
    
    def train_dataloader(self):
        return DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size,sampler=config['sampler']) 
    
    def val_dataloader(self):
        return DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size) 

    def test_dataloader(self):
        return DataLoader(test_dataset, shuffle=True, batch_size=1) 
    
    def configure_optimizers(self):    
        opt = torch_optimizer.Yogi(self.net.parameters(), lr=self.learning_rate)
#         opt = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=0, verbose=True)
        return [opt], [sch] 
    
    
trainer = pl.Trainer(max_epochs=config['epoch'],
                     resume_from_checkpoint=config['resume_from_checkpoint'],
                     logger = logger,
#                      callbacks=[metrics_callback],
                     checkpoint_callback=checkpoint_callback,
                     amp_level=config['amp_level'],
                     precision=config['precision'],
                     distributed_backend=config['distributed_backend'],
                     gpus=config['gpus'],             
                     auto_scale_batch_size =config['auto_scale_batch_size']
                    )
```

```

# Albumentation
```
# !pip install -U git+https://github.com/albu/albumentations
import albumentations as albu

def augmentation_train():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.MultiplicativeNoise(multiplier=(0.98, 1.02), per_channel=True, p=0.4),
        albu.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3),contrast_limit=(-0.3, 0.3),brightness_by_max=True, p=0.5),
        albu.RandomGamma(gamma_limit=(80,120), p=0.5),
        
        albu.OneOf(
        [albu.ElasticTransform(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,alpha=1,sigma=25,alpha_affine=25, p=0.5),
         albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-0.3,0.3),num_steps=5, p=0.5),
         albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-.05,.05),shift_limit=(-0.1,0.1), p=0.5),
        ]
        ,p=0.5),
                
        albu.OneOf([            
        albu.IAASharpen(alpha=(0,0.1), lightness=(0.01,0.03), p=0.3),
        albu.MotionBlur(blur_limit=(5), p=0.1),
        albu.GaussianBlur(blur_limit=(5), p=0.1),
        ],p=0.5),
        
        albu.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, p=0.5),
        albu.RandomCrop(height=448, width=448, always_apply=True), 

    ]
    return albu.Compose(train_transform)

def augmentation_valid(center=None):
    test_transform = []
    return albu.Compose(test_transform)
```

# Define modules
```
model = SegModel_plain(config)
trainer.fit(model)
```
