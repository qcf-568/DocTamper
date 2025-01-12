import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

# prepare: OCR pickle ocr.pk # {'img_path1':[[x1,y1,w1,h1], [x1,y1,w1,h1]], 'img_path2':[[x1,y1,w1,h1], [x1,y1,w1,h1]]}
# e.g. from paddleOCRv3 
# prepare: image_mask # 'imgs/0.jpg' -> 'masks/0.png'
# e.g. from image binarization

### OCR
# http://ocr.space/
# https://github.com/ErikBoesen/ocrspace

### Image Binarization
# import doxapy (pip install doxapy)
# import numpy as np
# model = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
# model.initialize(img)
# msk = np.zeros_like(img)
# model.to_binary(msk, {"window": 75, "k": 0.2})
# return msk

# run this script with "python stg1.py"

with open('ocr.pk','rb') as f: # OCR pickle {'img_path1':[[x1,y1,w1,h1], [x1,y1,w1,h1]], 'img_path2':[[x1,y1,w1,h1], [x1,y1,w1,h1]]}
    fpk = pickle.load(f)

img_cnt = 0
max_cnt = 10

def getdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

getdir('tamp_imgs')
getdir('tamp_masks')

for k, box in tqdm(fpk.items()):
	img1 = cv2.imread(k)
	h, w = img1.shape[:2]
	img2 = cv2.imread(k)
	mask = (cv2.imread(k.replace('imgs','masks')[:-4]+'.png', 0)>127)
	not_mask = np.logical_not(mask)
	gt = np.zeros((h, w), dtype=np.uint8)
	cnt = 0
	for bi1,b1 in enumerate(box):
		tgt = img1[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
		msk = mask[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
		not_msk = not_mask[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
		tgtfm = tgt[msk].mean().astype(np.float32)
		tgtfs = tgt[msk].std().astype(np.float32)
		tgtbm = tgt[not_msk].mean().astype(np.float32)
		tgtbs = tgt[not_msk].std().astype(np.float32)
		for bi2,b2 in enumerate(box):
			if bi1!=bi2:
				w11 = (0.9*b2[2])
				w12 = (1.1*b2[2])
				h11 = (0.9*b2[3])
				h12 = (1.1*b2[3])
				tgtfm2 = tgt[msk].mean().astype(np.float32)
				tgtfs2 = tgt[msk].std().astype(np.float32)
				tgtbm2 = tgt[not_msk].mean().astype(np.float32)
				tgtbs2 = tgt[not_msk].std().astype(np.float32)
				tgtfm21 = tgtfm2-20
				tgtfm22 = tgtfm2+20
				tgtfs21 = tgtfs2-4
				tgtfs22 = tgtfs2+4
				tgtbm21 = tgtbm2-20
				tgtbm22 = tgtbm2+20
				tgtbs21 = tgtbs2-4
				tgtbs22 = tgtbs2+4
				if (w11<=b1[2]<=w12) and (h11<=b1[3]<=h12) and (tgtfm21<=tgtfm<=tgtfm22) and (tgtbm21<=tgtbm<=tgtbm22) and (tgtfs21<=tgtfs<=tgtfs22) and (tgtbs21<=tgtbs<=tgtbs22):
					print(img1[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]].shape, img2[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]].shape)
					img1[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]] = cv2.resize(img2[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]], (int(b1[2]), int(b1[3])))
					gt[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]] = 255
					cnt = (cnt + 1)
					if cnt>max_cnt:
						cv2.imwrite('tamp_imgs/%d.jpg'%img_cnt, img1)
						cv2.imwrite('tamp_masks/%d.png'%img_cnt, gt)
						cnt = 0
						img_cnt = (img_cnt + 1)
						img1 = cv2.imread(k)
						gt = np.zeros((h, w), dtype=np.uint8)
	cv2.imwrite('tamp_imgs/%d.jpg'%img_cnt, img1)
	cv2.imwrite('tamp_masks/%d.png'%img_cnt, gt)
	cnt = 0
	img_cnt = (img_cnt + 1)
