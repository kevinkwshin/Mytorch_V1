import SimpleITK as sitk
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from Keeplearning.image_processing import *

image_depth,image_height,image_width = 160,160,160

class Dataset_3D(DataLoader):
    def __init__(self, x_list, y_list):
        'Initialization'
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x_list)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        x_idx = self.x_list[index]
        y_idx = self.y_list[index]
        
        # Load data and get label
        image_ = sitk.GetArrayFromImage(sitk.ReadImage(x_idx)).astype('float32')
        mask_ = sitk.GetArrayFromImage(sitk.ReadImage(y_idx)).astype('float32')

        # augmentation
        
#         image_ = sitk.ReadImage(x_idx)
#         tx = augmentation_bspline_tranform_parameter(image_)
#         image_ = augmentation_bspline_tranform(image_,tx,sitk.sitkBSpline)
#         image_ = sitk.GetArrayFromImage(image_)
        
#         mask_ = sitk.ReadImage(y_idx)
#         mask_ = augmentation_bspline_tranform(mask_,tx,sitk.sitkNearestNeighbor)
#         mask_ = sitk.GetArrayFromImage(mask_)
        
        image_header = sitk.ReadImage(x_idx) # 돌아간거 보정...
        original_shape = image_header.GetSize()
        orientation = image_header.GetDirection()
        
        image = image_resize_3D(image_,image_depth,image_height,image_width)
        image = image_preprocess_float(image)
        mask = image_resize_3D(mask_,image_depth,image_height,image_width,mode='nearest')
        
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        mask[mask!=0]=1
        
        image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
        mask = np.reshape(mask,(1,mask.shape[0],mask.shape[1],mask.shape[2]))
                                       
        filename = x_idx.split('/')[-1].split('.')[0]
        return {'image':image, 'mask':mask, 'original_shape':original_shape, 'filename':filename}
