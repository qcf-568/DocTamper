# convert the .mdb data to images, output in (args.input)_new dir.
import os
import cv2
import six
import lmdb
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='DocTamperV1-FCD')  ## path to the dataset dir, which contains data.mdb and lock.mdb
args = parser.parse_args()
a =  lmdb.open(args.input,readonly=True,lock=False,readahead=False,meminit=False)

def getdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

image_dir = args.input+'_new/images/'
mask_dir = args.input+'_new/masks/'
getdir(image_dir)
getdir(mask_dir)

with a.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode('utf-8')))
    for index in tqdm(range(len(nSamples))):
        img_key = 'image-%09d' % index
        imgbuf = txn.get(img_key.encode('utf-8'))
        with open(image_dir+'%d.jpg'%index, 'wb') as f:
            f.write(imgbuf)
        lbl_key = 'label-%09d' % index
        lblbuf = txn.get(lbl_key.encode('utf-8'))
        mask = cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)
        if mask.max()==1:
            mask=mask*255
        cv2.imwrite(mask_dir+'%d.png'%index, mask)
