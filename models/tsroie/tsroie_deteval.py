import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from ic15metric import DetectionIoUEvaluator

IC15Metric = DetectionIoUEvaluator()
ic15_results = []

def center_split(array,h):
    center = len(array)//2
    argmax = np.argmax(array)
    if ((center-h//8)<=argmax<=(center+h//8)):
        return True
    else:
        return False

def get_gt_boxes(ipt):
    boxes = []
    for l in ipt:
        box = l.rstrip().split(',')
        if box[-1]=='2':
            boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return boxes

def np2ic15(ipt):
    results = []
    for (y1, x1, y2, x2) in ipt:
        results.append(
            {
              'points': [(y1, x1), (y1, x2), (y2, x2), (y2, x1)],
              'ignore':  False,
            }
        )
    return results


kernel = np.ones((5,5),dtype=np.uint8)
if not os.path.exists('error_analysis'):
    os.mkdir('error_analysis')

for fi,files in enumerate(tqdm(os.listdir('sroie_pred'))):
    img = cv2.imread('sroie_pred/'+files,0)
    h, w = img.shape

    ### masks to boxes
    img = (img>224).astype(np.uint8)*255
    img = cv2.dilate(cv2.erode(img,kernel),kernel)
    x1s = []
    x2s = []
    y1s = []
    y2s = []
    rights = []
    lefts = []
    heights = []
    a,b = cv2.connectedComponents(img)
    for i in range(1,a):
        m = (b==i)
        xnz,ynz = m.nonzero()
        xmin = xnz.min()
        xmax = (xnz.max()+1)
        ymin = ynz.min()
        ymax = (ynz.max()+1)
        if (((xmax-xmin)<(h/200)) or ((xmax-xmin)*(ymax-ymin))<32):
            img[m]=0
        r = (int((ymax-ymin)*0.3)//2*2+1)
        k = np.ones((1,r),dtype=np.uint8)
        img[xmin:xmax,ymin:ymax] = cv2.dilate(cv2.erode(img[xmin:xmax,ymin:ymax],k),k)
    a,b = cv2.connectedComponents(img)
    for i in range(1,a):
        m = (b==i)
        xnz,ynz = m.nonzero()
        xmin = xnz.min()
        xmax = (xnz.max()+1)
        ymin = ynz.min()
        ymax = (ynz.max()+1)
        lefts.append(ymin)
        rights.append(ymax)
        heights.append((xmin+xmax))
        x1s.append(xmin)
        x2s.append(xmax)
        y1s.append(ymin)
        y2s.append(ymax)
    x1s = np.array(x1s,dtype=np.int32)
    x2s = np.array(x2s,dtype=np.int32)
    y1s = np.array(y1s,dtype=np.int32)
    y2s = np.array(y2s,dtype=np.int32)
    lefts = np.array(lefts)
    rights = np.array(rights)
    heights = np.array(heights)
    diffs = (np.abs(rights[None]-lefts[:,None])<10)
    hiffs = (np.abs(heights[None]-heights[:,None])<16)
    trius = (1-np.eye(len(rights))).astype(np.bool)
    uses = np.bitwise_and(np.bitwise_and(diffs,hiffs),trius)
    xnz,ynz = uses.nonzero()
    if (len(xnz)!=0):
        x0 = xnz[0]
        y0 = ynz[0]
        img[x1s[x0]:x2s[x0],rights[y0]-8:lefts[x0]+8]=255

    pred_boxes = []
    statics = []
    a,b = cv2.connectedComponents(img)
    for i in range(1,a):
        m = (b==i)
        xnz,ynz = m.nonzero()
        xmin = xnz.min()
        xmax = (xnz.max()+1)
        ymin = ynz.min()
        ymax = (ynz.max()+1)
        pred_boxes.append((ymin,xmin,ymax,xmax))
        w = (ymax-ymin)
        h = (xmax-xmin)
        statics.append(((w*h),(w/h),(xmax-xmin)))
    statics = np.array(statics)
    hsort = np.argsort(statics[:,2])
    if ((len(statics)>1) and ((statics[hsort[-1],1]<2.5) and (np.argmax(statics[:,0])==hsort[-1]) and (statics[hsort[-1],2]>(1.2*statics[hsort[-2],2])))):
        y1,x1,y2,x2 = pred_boxes[hsort[-1]]
        h = (x2-x1)
        w = (y2-y1)
        if img[x1:x2,y1:y2].mean()<250:
            diff = np.abs((np.diff(img[x1+(h//8):x2-(h//8),y1:y2].mean(1))/w))
            if (diff.max()>0.02):
                if center_split(diff, h):
                    img[(x1+x2)//2,y1:y2]=0
                else:
                    if (diff.max()>0.1):
                        img[x1+(x2-x1)//3,y1:y2]=0
                        img[x1+(x2-x1)//3*2,y1:y2]=0

    ### caculating scores
    pred_boxes = []
    a,b = cv2.connectedComponents(img)
    for i in range(1,a):
        m = (b==i)
        xnz,ynz = m.nonzero()
        xmin = xnz.min()
        xmax = (xnz.max()+1)
        ymin = ynz.min()
        ymax = (ynz.max()+1)
        pred_boxes.append((ymin,xmin,ymax,xmax))
    with open('test_txts/'+files[:-4]+'.txt') as f:
        fl = f.readlines()
    gt_boxes = get_gt_boxes(fl)
    gt_boxes_ic15 = np2ic15(gt_boxes)
    pred_boxes_ic15 = np2ic15(pred_boxes)
    image_metric = IC15Metric.evaluate_image(gt_boxes_ic15, pred_boxes_ic15)
    ic15_results.append(image_metric)

    ### error analysis
    if image_metric['hmean']!=1:
        img = np.stack((img,img,img),2)
        for gtb in gt_boxes:
            cv2.rectangle(img,(gtb[0],gtb[1]),(gtb[2],gtb[3]),(255,0,0),2)
        for pdb in pred_boxes:
            cv2.rectangle(img,(pdb[0],pdb[1]),(pdb[2],pdb[3]),(0,0,255),2)
        cv2.imwrite('error_analysis/'+files,img)


final_result = IC15Metric.combine_results(ic15_results)
print(final_result)

