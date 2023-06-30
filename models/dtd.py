import os
import cv2
import lmdb
import torch
import jpegio
import numpy as np
import torch.nn as nn
import gc
import math
import time
import copy
import logging
import torch.optim as optim
import torch.distributed as dist
import random
import pickle
import six
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss
from fph import FPH
import albumentations as A
from swins import *
from albumentations.pytorch import ToTensorV2
import torchvision
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from segmentation_models_pytorch.base import modules as md
from typing import Optional, Union, List
from segmentation_models_pytorch.base import SegmentationModel

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        ipt = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = ipt + self.drop_path(x)
        return x

class AddCoords(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r
    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim,dtype=input_tensor.dtype), torch.arange(y_dim,dtype=input_tensor.dtype))
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size,1,x_dim,y_dim)
        yy_c = yy_c.expand(batch_size,1,x_dim,y_dim)
        ret = torch.cat((input_tensor,xx_c,yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret

class VPH(nn.Module):
    def __init__(self, dims=[96, 192], drop_path_rate=0.4, layer_scale_init_value=1e-6):
        super().__init__()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(6, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0], eps=1e-6, data_format="channels_first")), nn.Sequential(LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))])
        self.stages = nn.ModuleList([nn.Sequential(*[ConvBlock(dim=dims[0], drop_path=dp_rates[j],layer_scale_init_value=layer_scale_init_value) for j in range(3)]), nn.Sequential(*[ConvBlock(dim=dims[1], drop_path=dp_rates[3 + j],layer_scale_init_value=layer_scale_init_value) for j in range(3)])])
        self.apply(self._init_weights)

    def initnorm(self):
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(self.dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x):
        outs = []
        x = self.stages[0](self.downsample_layers[0](x))
        outs = [self.norm0(x)]
        x = self.stages[1](self.downsample_layers[1](x))
        outs.append(self.norm1(x))
        return outs

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class DecoderBlock(nn.Module):
    def __init__(self,cin,cadd,cout,):
        super().__init__()
        self.cin = (cin + cadd)
        self.cout = cout
        self.conv1 = md.Conv2dReLU(self.cin,self.cout,kernel_size=3,padding=1,use_batchnorm=True)
        self.conv2 = md.Conv2dReLU(self.cout,self.cout,kernel_size=3,padding=1,use_batchnorm=True)

    def forward(self, x1, x2=None):
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:,:self.cin])
        x1 = self.conv2(x1)
        return x1

class ConvBNReLU(nn.Module):
    def __init__(self,in_c,out_c,ks,stride=1,norm=True,res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False)
        self.res = res
    def forward(self,x):
        if self.res:
            return (x + self.conv(x))
        else:
            return self.conv(x)

class FUSE1(nn.Module):
    def __init__(self,in_channels_list=(96,192,384,768)):
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2],in_channels_list[2],1)
        self.c32 = ConvBNReLU(in_channels_list[3],in_channels_list[2],1)
        self.c33 = ConvBNReLU(in_channels_list[2],in_channels_list[2],3)

        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1,x2,x3 = x
        h,w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3),size=(h,w))+self.c31(x2))
        h,w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2),size=(h,w))+self.c21(x1))
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w))+self.c11(x))
        return x,x1,x2,x3

class FUSE2(nn.Module):
    def __init__(self,in_channels_list=(96,192,384)):
        super(FUSE2, self).__init__()

        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1,x2 = x
        h,w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2),size=(h,w),mode='bilinear',align_corners=True)+self.c21(x1))
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x))
        return x,x1,x2

class FUSE3(nn.Module):
    def __init__(self,in_channels_list=(96,192)):
        super(FUSE3, self).__init__()

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1 = x
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x))
        return x,x1

class MID(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [96]
        self.out_channels = decoder_channels
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1])
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, *features):
        decoder_features = {}
        features = self.fuse1(features)[::-1]
        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0],features[1])
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1],features[2])
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2],features[3])
        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2((decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]))
        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](decoder_features["x_0_0"], torch.cat((decoder_features["x_1_1"], features[2]),1))
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](decoder_features["x_1_1"], torch.cat((decoder_features["x_2_2"], features[3]),1))
        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3((decoder_features["x_1_2"], decoder_features["x_0_1"]))
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](decoder_features["x_0_1"], torch.cat((decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]),1))
        return self.decoder_convs["x_0_3"](torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"]),1))


class DTD(SegmentationModel):
    def __init__(self, encoder_name = "resnet18", decoder_channels = (384, 192, 96, 64), classes = 1):
        super().__init__()
        self.vph = torch.load('vph_imagenet.pt')
        self.swin = torch.load('swin_imagenet.pt')
        self.fph = FPH()
        self.decoder = MID(encoder_channels=(96, 192, 384, 768), decoder_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, upsampling=2.0)
        self.addcoords = AddCoords()
        self.FU = nn.Sequential(SCSEModule(448),nn.Conv2d(448,192,3,1,1),nn.BatchNorm2d(192),nn.ReLU(True))
        self.classification_head = None
        self.initialize()

    def forward(self,x,dct,qt):
        features = self.vph(self.addcoords(x))
        features[1] = self.FU(torch.cat((features[1],self.fph(dct,qt)),1))
        rst = self.swin[0](features[1].flatten(2).transpose(1,2).contiguous())
        N,L,C = rst.shape
        H = W = int(L**(1/2))
        features.append(self.vph.norm2(rst.transpose(1,2).contiguous().view(N,C,H,W)))
        features.append(self.vph.norm3(self.swin[2](self.swin[1](rst)).transpose(1,2).contiguous().view(N,C*2,H//2,W//2)))
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)

class seg_dtd(nn.Module):
    def __init__(self, model_name='resnet18', n_class=1):
        super().__init__()
        self.model = DTD(encoder_name=model_name, classes=n_class)

    @autocast()
    def forward(self, x, dct, qt):
        x = self.model(x, dct, qt)
        return x

