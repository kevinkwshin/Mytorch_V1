import scipy
from scipy import ndimage
import numpy as np
import os, glob, re

import cv2
import SimpleITK as sitk
import nibabel as nib

import matplotlib.pyplot as plt

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

from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings('ignore')

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

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