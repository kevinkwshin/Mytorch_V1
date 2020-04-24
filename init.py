import scipy
from scipy import ndimage
import numpy as np
import os, glob, re

import cv2
import SimpleITK as sitk
import nibabel as nib

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch import autograd, optim
from torch.nn import Module
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print()
print('torch.__version__ :', torch.__version__)
print('torch.cuda.is_available() :', torch.cuda.is_available())
print('torch.cuda.device_count() :', torch.cuda.device_count())

def model_test(model,inputs_shape):
    model.eval()
    try:
        inputs = torch.rand(inputs_shape)
        preds = model(inputs)
        print('inputs_shape',inputs_shape)
        print('shape:',preds.shape)
        print('preds',preds)
    except:
        inputs = torch.rand(inputs_shape).cuda()
        preds = model(inputs).cuda()
        print('inputs_shape',inputs_shape)
        print('shape:',preds.shape)
        print('preds',preds)
