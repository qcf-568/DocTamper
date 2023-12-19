import os
import cv2
import six
import lmdb
import argparse
from PIL import Image
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='DocTamperV1-FCD')  ## path to the dataset dir, which contains data.mdb and lock.mdb
parser.add_argument('--i', type=int, default=0) # index of the image to be visualized
args = parser.parse_args()
a =  lmdb.open(args.input,readonly=True,lock=False,readahead=False,meminit=False)
index = args.i
with a.begin(write=False) as txn:
    img_key = 'image-%09d' % index
    imgbuf = txn.get(img_key.encode('utf-8'))
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf)
    im.save('a.jpg')
    lbl_key = 'label-%09d' % index
    lblbuf = txn.get(lbl_key.encode('utf-8'))
    mask = cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)
    print(mask.max())
    if mask.max()==1:
        mask=mask*255
    cv2.imwrite('a.png',mask)

