import os
import cv2
import json
import imagesize
import numpy as np
filenames = {}

os.mkdir('test_masks')
os.mkdir('train_masks')

with open('sroie_train_1011.json') as f:
    js = json.load(f)

for i in range(len(js['images'])):
    filename = js['images'][i]['file_name']
    this_id = js['images'][i]['id']
    filenames[this_id] = {'filename':filename,'polys':[]}

for i in range(len(js['annotations'])):
    if js['annotations'][i]['category_id'] == 2:
        this_id = js['annotations'][i]['image_id']
        seg1 = np.array(js['annotations'][i]['segmentation'],dtype=np.int32).reshape(4,2)
        filenames[this_id]['polys'].append(seg1)

for v in filenames.values():
    w,h = imagesize.get('train/'+v['filename'])
    mask1 = np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask1,v['polys'],255)
    cv2.imwrite('train_masks/%s.png'%v['filename'][:-4],mask1)

filenames = {}
with open('sroie_test_1011.json') as f:
    js = json.load(f)

for i in range(len(js['images'])):
    filename = js['images'][i]['file_name']
    this_id = js['images'][i]['id']
    filenames[this_id] = {'filename':filename,'polys':[]}

for i in range(len(js['annotations'])):
    if js['annotations'][i]['category_id'] == 2:
        this_id = js['annotations'][i]['image_id']
        seg1 = np.array(js['annotations'][i]['segmentation'],dtype=np.int32).reshape(4,2)
        filenames[this_id]['polys'].append(seg1)

for v in filenames.values():
    w,h = imagesize.get('test/'+v['filename'])
    mask1 = np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask1,v['polys'],255)
    cv2.imwrite('test_masks/%s.png'%v['filename'][:-4],mask1)
