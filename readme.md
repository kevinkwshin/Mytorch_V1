* To release memory, torch.cuda.empty_cache(). But it would be better to reset ur kernel.

https://github.com/BloodAxe/pytorch-toolbelt

# Import Library
```
!nvidia-smi

import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2";

import torch
gpus = torch.cuda.device_count()
print(torch.cuda.is_available())
print('available gpu:',gpus)

import Mytorch
from Mytorch.losses import *
```

# Parameter Setting & Research Note
```
task_name = 
loss = 
filename = 
gpus = 

# 1st Try
Focal Loss + Model1
# 2nd Try
Focal Loss + Model2

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
```
batch_size = 20 # 12  # 24g당 12개
# from Mytorch.imbalanced import ImbalancedDatasetSampler

# def for_sampler_get_label(dataset, idx):
#     return int(dataset[idx][2])

# train_loader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=0, drop_last=True, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=for_sampler_get_label))
# valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

train_dataset = Dataset_npy(x_train,y_train,
                            augmentation=augmentation_train(),
                            )

valid_dataset = Dataset_npy(x_valid,y_valid,
                            augmentation=augmentation_valid(),
                            )

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          num_workers=2,
                          shuffle=True,
#                           sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=for_sampler_get_label)                          
#                           sampler=torch.utils.data.SubsetRandomSampler(range(int(len(train_dataset)*0.2))), shuffle=False, # boost train speed
                         )

import pickle
batch_size = 1
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_size,
                          num_workers=2,
                          shuffle=True,
                         )
```

# Albumentation
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
# Define Model
```
import Mytorch as smp

model = smp.Unet(encoder_name='timm-efficientnet-b7',
                 decoder_attention_type='scse',        
                 classes=5,
                 encoder_weights=None,
                 activation='softmax',)
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
  
    try:
        score.append(valid_logs['loss_main']+valid_logs['loss_aux'])
        if np.min(score) == valid_logs['loss_main']+valid_logs['loss_aux'] and epoch > 5:
            model_save_state_dict(model,filename,parallel_mode)
    except:
        score.append(valid_logs['loss_main'])
        if np.min(score) == valid_logs['loss_main'] and epoch > 5:
            model_save_state_dict(model,filename,parallel_mode)

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
