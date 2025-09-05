import os
import cv2
import torch#用户ID：7fb702cd-1293-4470-a3b2-4ba88c3b3d4a
import numpy as np
import pickle
import torch.nn as nn
import gc
import math
import time
import copy
import jpegio
import logging
import tempfile
import torch.optim as optim
import torch.distributed as dist
import random
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dtd import *
import argparse
import torchvision
parser = argparse.ArgumentParser()
parser.add_argument('--train_imgs_dir', type=str, default='/home/jingroup/storage/chenfan/CVPR/datas/jpegc1_use')
parser.add_argument('--val_imgs_dir', type=str, default='/home/jingroup/storage/chenfan/CVPR/datas/jpegc1_use')
parser.add_argument('--train_labels_dir', type=str, default='/home/jingroup/storage/chenfan/CH_DOC/label')
parser.add_argument('--val_labels_dir', type=str, default='/home/jingroup/storage/chenfan/CH_DOC/val_label')
parser.add_argument('--model_name', type=str, default='efficientnet-b6')
parser.add_argument('--att', type=str, default='None')
parser.add_argument('--num', type=str, default='1')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--bs', type=int, default=12)
parser.add_argument('--base', type=int, default=1)
parser.add_argument('--lr_base', type=float, default=3e-4)
parser.add_argument('--cp', type=float, default=1.0)
parser.add_argument('--mode', type=str, default='0123')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--adds', type=str, default='123')
parser.add_argument('--lossw', type=str, default='1,2,3,4')

train_transform = A.Compose([
    # reszie
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5)
])

val_transform = A.Compose([
    A.Resize(512, 512),
    # A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    # ToTensorV2(),
])

infer_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5,0.5)),
    ToTensorV2(),
])

from copy import deepcopy

class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.cnts = 0
        self.mode = (not ('test' in imgs_dir))
        if self.mode:
            with open('../sroie/train.pk','rb') as fpk:
                self.fpk = pickle.load(fpk)
        else:
            with open('../sroie/test.pk','rb') as fpk:
                self.fpk = pickle.load(fpk)
        print('*'*60)
        print('DCT inited!')
        print('*'*60)
        self.transform = transform
        self.imgs = os.listdir(self.imgs_dir)
        with open('qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.Tensor(v)
        with open('qtm75.pk','rb') as fpk:
            self.pkf = pickle.load(fpk)
            self.pkfkeys = self.pkf.keys()
        self.qs = np.arange(90,101)
        # with open('qt75.pk','rb') as f:
        #     self.qt75 = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])
        logging.info(f'Creating dataset with {len(self.imgs)} examples',self.mode)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        imgi = self.imgs[i]
        img_file = self.imgs_dir + '/' + imgi
        mask_file = self.masks_dir + '/' + imgi[:-4] + '.png'
        image = np.array(Image.open(img_file))
        mask = (cv2.imread(mask_file, 0)>0).astype(np.uint8)
        H,W = mask.shape[:2]
        if H < 512:
            dh = (512-H)
        else:
            dh = 0
        if W < 512:
            dw = (512-W)
        else:
            dw = 0
        mask = np.pad(mask,((0,dh),(0,dw)),'constant',constant_values=0)
        image = np.pad(image,((0,dh),(0,dw),(0,0)),'constant',constant_values=255)
        H,W = mask.shape
        H864 = H//8-64
        W864 = W//8-64
        if random.uniform(0,1)<0.1:
          if (len(self.fpk[imgi])!=0):
            sxu,syu = random.choice(self.fpk[imgi])
          else:
            H,W = mask.shape[:2]
            sxu = random.randint(0,H864)
            syu = random.randint(0,W864)
        elif random.uniform(0,1)<0.2:
          for t in range(4):
            sxu = random.randint(0,H864)*8
            syu = random.randint(0,W864)*8
            mask_ = mask[sxu:(sxu+512),syu:(syu+512)]
            if mask_.max() != 0:
                break
        else:
            sxu = random.randint(0,H864)*8
            syu = random.randint(0,W864)*8
            if random.uniform(0,1)<0.15:
                sxu = 0
            if random.uniform(0,1)<0.15:
                syu = 0
        image = image[sxu:sxu+512,syu:syu+512]
        mask = mask[sxu:sxu+512,syu:syu+512]
        if (self.mode and (random.uniform(0,1)<0.3)):
            qu = random.randint(2,12)
            while not (qu in self.pkfkeys):
                qu = random.randint(2,12)
            qu2 = qu+random.randint(1,24)
            while not (qu2 in self.pkfkeys):
                qu2 = qu+random.randint(1,24)
            image = Image.fromarray(image).convert('L')
            image2 = deepcopy(image)
            # print('1',random.choice(self.pkf[qu]))
            # print('2',random.choice(self.pkf[qu2]))
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                image.save(tmp,"JPEG",qtables={0:random.choice(self.pkf[qu])})
                image = np.array(Image.open(tmp))
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                image2.save(tmp,"JPEG",qtables={0:random.choice(self.pkf[qu2])})
                image2 = np.array(Image.open(tmp))
            im = Image.fromarray(image*mask+image2*(1-mask)).convert('RGB')
            # qu = random.randint(90,99)
            # imt = cv2.imdecode(cv2.imencode('.jpg',image,[1,qu])[1],1) 
            # imt2 = cv2.imdecode(cv2.imencode('.jpg',image,[1,qu-random.randint(1,20)])[1],1)
            # image = imt*mask[...,None]+imt2*(1-mask[...,None])
        else:
            im = Image.fromarray(image)
        masksum = mask.sum()
        if masksum<8:
            cls_lbl = 0
        elif masksum>512:
            cls_lbl = 1
        else:
            cls_lbl = -1
        mask = self.totsr(image=mask.copy())['image']
        q1,q2 = np.random.choice(self.qs,2,replace=False)
        q1 = int(q1)
        q2 = int(q2)
        use_qt = torch.stack((self.pks[q1],self.pks[q2]))
        sidx_temp = torch.stack([torch.randperm(2) for i in range(64)],1).reshape(2,8,8)
        new_qt = [nqt.short().flatten().tolist() for nqt in torch.gather(use_qt,0,sidx_temp)]
        new_qtb = {0:[int(x) for x in new_qt[0]]}
        if True:
                if random.uniform(0,1) < 0.5:
                    im = self.hflip(im)
                    mask = self.hflip(mask)
                if random.uniform(0,1) < 0.5:
                    im = self.vflip(im)
                    mask = self.vflip(mask)
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    im = im.convert("L")
                    im.save(tmp,"JPEG",qtables=new_qtb)
                    jpg = jpegio.read(tmp.name)
                    dct = jpg.coef_arrays[0].copy()
                    im = im.convert('RGB')
                return {
                    'image': self.toctsr(im),
                    'label': mask.long(),
                    'rgb': np.clip(np.abs(dct),0,20),
                    'cls_lbl': cls_lbl,
                    'q': torch.LongTensor(new_qt[0]),
                }
train_data = RSCDataset('../sroie/train', '../sroie/train_masks', transform=train_transform)
valid_data = RSCDataset('../sroie/test', '../sroie/test_masks', transform=val_transform)


args = parser.parse_args()
ngpu = torch.cuda.device_count()
ngpub = ngpu * args.base
if ngpu > 1:
    gpus = True
    device = torch.device("cuda",args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
else:
    gpus = False
    device = torch.device("cuda")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))

def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from mmseg.utils import get_root_logger
from torch.nn import functional as F
from segmentation_models_pytorch.base import modules as md
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    ClassificationHead,
)
from segmentation_models_pytorch.base import modules as md

model=seg_dtd(args.model_name,args.n_class).to(device)
if gpus:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model= torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
model_name = args.model_name
save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', model_name)
try:
  if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
except:
  pass
try:
  if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)
except:
  pass

# ngpub = max(ngpub,2)
# 参数设置
param = {}
param['batch_size'] = args.bs       # 批大小
param['epochs'] = 541       # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 4       # 保存间隔(epoch)
param['iter_inter'] = 64     # 显示迭代间隔(batch)
param['min_inter'] = 10
param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0']=int(24/ngpub)  #cosine warmup的参数
# param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92]}
param['load_ckpt_dir'] = None

def train_net_qyl(param, model, train_data, valid_data, plot=False,device='cuda'):
    # 初始化参数
    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    iter_inter      = param['iter_inter']
    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']
    T0=param['T0']
    scaler = GradScaler()
    if gpus:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    lr_base = args.lr_base 
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    if gpus:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8, sampler=train_sampler)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, num_workers=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4 ,weight_decay=5e-4)
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
    #optimizer=Ranger(model.parameters(),lr=1e-3) from .ranger import Ranger
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    iter_per_epoch = len(train_loader)
    totalstep = epochs*iter_per_epoch
    warmupr = 1/epochs
    warmstep = 20
    lr_min = 1e-5
    lr_min /= lr_base
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x: ((((1+math.cos((x-warmstep)*math.pi/(totalstep-warmstep)))/2)+lr_min) if (x > warmstep) else (x/warmstep+lr_min)))
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2, eta_min=1e-6, last_epoch=-1)
    #scheduler=ShopeeScheduler(optimizer,**scheduler_params) from .custom_lr import ShopeeScheduler
    #criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    # LovaszLoss_fn=LovaszLoss(mode='multiclass')
    dl = DiceLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.001)
    #logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) +'_'+model_name+ '.log'))
    logger = get_logger(os.path.join(save_log_dir, time.strftime("%m-%d", time.localtime()) +'_'+model_name+ '.log'))
    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_iou = 0
    best_epoch=0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    lossw = [float(lw) for lw in args.lossw.split(',')]
    lossws = sum(lossw)
    if False:#load_ckpt_dir is not None:
        ckpt = torch.load('checkpoint-best.pth')
        model.load_state_dict(ckpt['state_dict'])
    logger.info('Total Epoch:{} Training num:{}  Validation num:{}'.format(epochs, train_data_size, valid_data_size))
    celossf = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target, dct, qs, cls_lbl = batch_samples['image'], batch_samples['label'], batch_samples['rgb'], batch_samples['q'], batch_samples['cls_lbl']
            data, target, dct, qs, cls_lbl = Variable(data.to(device)), Variable(target.to(device)), Variable(dct.to(device)), Variable(qs.to(device)), Variable(cls_lbl.to(device))
            qs = qs.reshape(-1,1,8,8)
            # print(data.shape)
            pred = model(data,dct,qs)
            loss = dl(pred, target)+SoftCrossEntropy_fn(pred, target)
            loss.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()
        #scheduler.step()
        # 验证阶段
        if (epoch%6!=0):
            continue
        model.eval()
        print('eval start')
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        iou=IOUMetric(2)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target, dct, qs, cls_lbl = batch_samples['image'], batch_samples['label'], batch_samples['rgb'], batch_samples['q'], batch_samples['cls_lbl']
                data, target, dct, qs, cls_lbl = Variable(data.to(device)), Variable(target.to(device)), Variable(dct.to(device)), Variable(qs.to(device)), Variable(cls_lbl.to(device))
                qs = qs.reshape(-1,1,8,8)
                pred = model(data,dct,qs)
                loss = SoftCrossEntropy_fn(pred, target)
                pred=pred.cpu().data.numpy()
                pred= np.argmax(pred,axis=1)
                iou.add_batch(pred,target.cpu().data.numpy())
                #
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
                # if batch_idx % iter_inter == 0:
                #     logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                #         epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, valid_iter_loss.avg))
            val_loss=valid_iter_loss.avg
            acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
            logger.info('[val] epoch:{} iou:{}'.format(epoch,iu))
                

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        # if epoch in save_epoch[T0]:
        #     torch.save(model.state_dict(),'{}/cosine_epoch{}.pth'.format(save_ckpt_dir,epoch))
        # state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        # filename = os.path.join(save_ckpt_dir, 'checkpoint-latest.pth')
        # torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if iu[1] > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = iu[1]
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
        #scheduler.step()
            
    return #best_mode, model

train_net_qyl(param, model, train_data, valid_data, device=device)

