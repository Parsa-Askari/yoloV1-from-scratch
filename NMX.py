from IOU import iou
import torch
import numpy as np
def nmx(boxes,iou_thresh=0.6,prob_thresh=0.5):
    """
        boxes : [S*S,class,prob,w_c,h_c,w,h]
    """
    boxes=sorted(boxes,key= lambda x : x[1] ,reverse=True)
    boxes=np.array(boxes)  
    boxes=boxes[boxes[:,1]>prob_thresh]
    new_boxes=[]
    """
    [[  5.    0.9  40.   40.   20.   20. ]
    [  1.    0.8 100.  100.   50.   50. ]
    [  3.    0.7 200.  100.   50.   50. ]
    [  4.    0.6 200.  300.   50.   50. ]]
    ++
    [  1.    0.9 100.  100.   50.   50. ]
    """
    while(boxes.size>0):
        refrence=boxes[0]
        boxes=boxes[1:]
        iou_scores=iou(refrence[2:],boxes[...,2:],True).numpy()
        cond=((boxes[...,0:1]!=refrence[0:1]) | (iou_scores<iou_thresh)).reshape(-1)
        boxes=boxes[cond]
        new_boxes.append(refrence.tolist())
    return new_boxes
# boxes=[[1,0.9,100,100,50,50],
#        [1,0.8,50,50,10,10],
#        [2,0.2,200,100,50,50],
#        [3,0.7,200,100,50,50],
#        [4,0.6,200,300,50,50],
#        [5,0.9,40,40,20,20]]
# boxes=nmx(boxes)
# print(boxes)