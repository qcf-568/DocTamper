import os
import cv2
import lmdb
import torch
import jpegio
import numpy as np


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


iou=IOUMetric(2)
precisons = []
recalls = []
with torch.no_grad():
    for batch_idx, batch_samples in enumerate(tqdm(test_loader)):
        pred = model(datas) # pred of shape (Batchsize, 2, img_Height, img_Width)
        pred_tamper = pred.argmax(1)
        target_ = target.squeeze(1)
        match = (pred_tamper*target_).sum((1,2))
        preds = pred_tamper.sum((1,2))
        target_sum = target_.sum((1,2))
        precisons.append((match/(preds+1e-8)).mean().item())
        recalls.append((match/target_sum).mean().item())
        pred=pred.cpu().data.numpy()
        pred= np.argmax(pred,axis=1)
        iou.add_batch(pred,target.cpu().data.numpy())
    acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
    precisons = np.array(precisons).mean()
    recalls = np.array(recalls).mean()
    print('[val] iou:{} p:{} r:{} f:{}'.format(iu[1],precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8))))


