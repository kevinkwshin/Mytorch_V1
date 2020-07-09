import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..base import modules as md
from ..base import init_weights


class base_PT(nn.Module):
    def __init__(
            self,
            pooling_scale,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(pooling_scale, pooling_scale, ceil_mode=True)
        self.conv2d = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class base_Cat(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.Cat_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.Cat_bn = nn.BatchNorm2d(out_channels)
        self.Cat_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Cat_conv(x)
        x = self.Cat_bn(x)
        x = self.Cat_relu(x)
        return x

class base_UT(nn.Module):
    def __init__(
            self,
            up_scale,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.UT_upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear')  # 14*14
        self.UT_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.UT_bn = nn.BatchNorm2d(out_channels)
        self.UT_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.UT_upsample(x)
        x = self.UT_conv(x)
        x = self.UT_bn(x)
        x = self.UT_relu(x)
        return x

class base_Fusion(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            attention_type=None,
    ):
        super().__init__()
        self.attention1 = md.Attention(attention_type, in_channels=in_channels)
        self.Fusion_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)  # 16
        self.Fusion_bn = nn.BatchNorm2d(out_channels)
        self.Fusion_relu = nn.ReLU(inplace=True)
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.attention1(x)
        x = self.Fusion_conv(x)
        x = self.Fusion_bn(x)
        x = self.Fusion_relu(x)
        x = self.attention2(x)
        return x

class base_LAST(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            attention_type=None,
    ):
        super().__init__()
        self.last_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.last_conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)  # 16
        self.last_bn1 = nn.BatchNorm2d(out_channels)
        self.last_relu1 = nn.ReLU(inplace=True)
        self.last_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)  # 16
        self.last_bn2 = nn.BatchNorm2d(out_channels)
        self.last_relu2 = nn.ReLU(inplace=True)
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.last_upsample(x)
        x = self.last_conv1(x)
        x = self.last_bn1(x)
        x = self.last_relu1(x)
        x = self.last_conv2(x)
        x = self.last_bn2(x)
        x = self.last_relu2(x)
        x = self.attention2(x)
        return x

################################################################################################################################################

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            decoder_type='upsample',
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1]) # [:-1] 는 마지막 인덱스 빼기.
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock( head_channels, head_channels, use_batchnorm=use_batchnorm )
        else:
            self.center = nn.Identity()   # efficient 이거로 들어감. 

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        ## -------------Decoder--------------
        # 순서가 디코더 단 4,3,2,1 순서임!!
        # in_channels, skip_channels, out_channels   
        # print("in_channels", in_channels)             in_channels [640, 256, 128, 64, 32]
        # print("skip_channels", skip_channels)         skip_channels [224, 80, 48, 64, 0]
        # print("out_channels", out_channels)           out_channels (256, 128, 64, 32, 16)
        # print("encoder_channels", encoder_channels)   encoder_channels (640, 224, 80, 48, 64)  # 실제 encoder 아웃풋.

        filters = encoder_channels[::-1]
        self.CatChannels = filters[0]  # 64
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks # 320

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4_module = base_PT(pooling_scale=8, in_channels=filters[0], out_channels=self.CatChannels)
        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4_module = base_PT(pooling_scale=4, in_channels=filters[1], out_channels=self.CatChannels)
        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4_module = base_PT(pooling_scale=2, in_channels=filters[2], out_channels=self.CatChannels)
        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_module = base_Cat(in_channels=filters[3], out_channels=self.CatChannels)
        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4_module = base_UT(up_scale=2, in_channels=filters[4], out_channels=self.CatChannels)
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.hd4_module = base_Fusion(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)
 
        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3_module = base_PT(pooling_scale=4, in_channels=filters[0], out_channels=self.CatChannels)
        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3_module = base_PT(pooling_scale=2, in_channels=filters[1], out_channels=self.CatChannels)
        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_module = base_Cat(in_channels=filters[2], out_channels=self.CatChannels)
        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3_module = base_UT(up_scale=2, in_channels=self.UpChannels, out_channels=self.CatChannels)
        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3_module = base_UT(up_scale=4, in_channels=filters[4], out_channels=self.CatChannels)
        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.hd3_module = base_Fusion(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2_module = base_PT(pooling_scale=2, in_channels=filters[0], out_channels=self.CatChannels)
        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_module = base_Cat(in_channels=filters[1], out_channels=self.CatChannels)
        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2_module = base_UT(up_scale=2, in_channels=self.UpChannels, out_channels=self.CatChannels)
        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2_module = base_UT(up_scale=4, in_channels=self.UpChannels, out_channels=self.CatChannels)
        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2_module = base_UT(up_scale=8, in_channels=filters[4], out_channels=self.CatChannels)
        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.hd2_module = base_Fusion(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_module = base_Cat(in_channels=filters[0], out_channels=self.CatChannels)
        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1_module = base_UT(up_scale=2, in_channels=self.UpChannels, out_channels=self.CatChannels)
        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1_module = base_UT(up_scale=4, in_channels=self.UpChannels, out_channels=self.CatChannels)
        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1_module = base_UT(up_scale=8, in_channels=self.UpChannels, out_channels=self.CatChannels)
        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1_module = base_UT(up_scale=16, in_channels=filters[4], out_channels=self.CatChannels)
        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.hd1_module = base_Fusion(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)

        '''Last'''
        self.last5 = base_LAST(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)
        self.last4 = base_LAST(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)
        self.last3 = base_LAST(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)
        self.last2 = base_LAST(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)
        self.last1 = base_LAST(in_channels=self.UpChannels, out_channels=self.UpChannels, attention_type=attention_type)

        # -------------Bilinear Upsampling--------------
        # self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # -------------DeepSup--------------
        self.outconv1 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)    # 1 = class 가 1개 이기에
        self.outconv2 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], 1, 3, padding=1)
        
        # -------------Auxiliary classifier--------------
        self.cls = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv2d(filters[4], 2, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights.init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights.init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, C, H, W = seg.size()
        seg = seg.view(B, C, H * W)
        final = torch.einsum("ijk, ij -> ijk", [seg, cls])          # 아인슈타인 표기법 벡터곱인듯
        final = final.view(B, C, H, W)
        return final

    def forward(self, *features):
        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        
        head = features[0]
        skips = features[1:] # [1:]는 제일 앞에 인덱스 빼고.

        # print("len(skips)", len(skips)) # 4
        # print("len(features)", len(features)) # 5
        # print("len(skips)", len(skips)) # 4
        # print("len(self.blocks", len(self.blocks)) # 5

        x = self.center(head)

        ## -------------Skip--------------
        skip_4 = skips[0]
        skip_3 = skips[1]
        skip_2 = skips[2]
        skip_1 = skips[3]
        x_center = x  # hd5 = x_center

        # -------------Classification-------------
        cls_branch = self.cls(x_center).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()

        ## -------------Decoder-------------
        # 디코더 4번 단
        h1_PT_hd4 = self.h1_PT_hd4_module(skip_1)
        h2_PT_hd4 = self.h2_PT_hd4_module(skip_2)
        h3_PT_hd4 = self.h3_PT_hd4_module(skip_3)
        h4_Cat_hd4 = self.h4_Cat_hd4_module(skip_4)
        hd5_UT_hd4 = self.hd5_UT_hd4_module(x_center)
        hd4 = self.hd4_module(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)) # hd4->40*40*UpChannels

        # print("x_center", x_center.shape) # x_center torch.Size([17, 640, 10, 10])
        # print("h1_PT_hd4", h1_PT_hd4.shape)
        # print("h2_PT_hd4", h2_PT_hd4.shape)
        # print("h3_PT_hd4", h3_PT_hd4.shape)
        # print("h4_Cat_hd4", h4_Cat_hd4.shape)
        # print("hd5_UT_hd4", hd5_UT_hd4.shape)

        # 디코더 3번 단
        h1_PT_hd3 = self.h1_PT_hd3_module(skip_1)
        h2_PT_hd3 = self.h2_PT_hd3_module(skip_2)
        h3_Cat_hd3 = self.h3_Cat_hd3_module(skip_3)
        hd4_UT_hd3 = self.hd4_UT_hd3_module(hd4)
        hd5_UT_hd3 = self.hd5_UT_hd3_module(x_center)
        hd3 = self.hd3_module(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)) # hd3->80*80*UpChannels
        
        # 디코더 2번 단
        h1_PT_hd2 = self.h1_PT_hd2_module(skip_1)
        h2_Cat_hd2 = self.h2_Cat_hd2_module(skip_2)
        hd3_UT_hd2 = self.hd3_UT_hd2_module(hd3)
        hd4_UT_hd2 = self.hd4_UT_hd2_module(hd4)
        hd5_UT_hd2 = self.hd5_UT_hd2_module(x_center)
        hd2 = self.hd2_module(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)) # hd2->160*160*UpChannels

        # 디코더 1번 단 (마지막 seg_head랑 연결되는 부분)
        h1_Cat_hd1 = self.h1_Cat_hd1_module(skip_1)
        hd2_UT_hd1 = self.hd2_UT_hd1_module(hd2)
        hd3_UT_hd1 = self.hd3_UT_hd1_module(hd3)
        hd4_UT_hd1 = self.hd4_UT_hd1_module(hd4)
        hd5_UT_hd1 = self.hd5_UT_hd1_module(x_center)
        hd1 = self.hd1_module(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)) # hd1->320*320*UpChannels

        # d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        # return F.sigmoid(d1)

        # x_center = self.last5(x_center)  # last 통과해야 320*320 가 된다.
        # hd4 = self.last4(hd4)  # last 통과해야 320*320 가 된다.
        # hd3 = self.last3(hd3)  # last 통과해야 320*320 가 된다.
        # hd2 = self.last2(hd2)  # last 통과해야 320*320 가 된다.
        hd1 = self.last1(hd1)  # last 통과해야 320*320 가 된다.


        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return F.sigmoid(d1)


