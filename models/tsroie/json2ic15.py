import os
import cv2
import json
import imagesize
import numpy as np
filenames = {}

if not os.path.exists('test_txts'):
    os.mkdir('test_txts')

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
        y1 = seg1[:,0].min()
        x1 = seg1[:,1].min()
        y2 = seg1[:,0].max()
        x2 = seg1[:,1].max()
        filenames[this_id]['polys'].append((y1,x1,y2,x2))

for v in filenames.values():
    vp = v['polys']
    with open('test_txts/%s.txt'%v['filename'][:-4],'w') as f:
        for vpp in vp:
            f.write('{},{},{},{},2\n'.format(vpp[0],vpp[1],vpp[2],vpp[3]))
