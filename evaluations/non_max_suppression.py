from .iou import calculate_iou
import numpy as np
def non_max_suppression(boxes,prob_thresh,iou_thresh):
    """
        boxes : bounding boxes . [object_prob,class,x,y,w,h]
        iou_thresh : a threshold for iou score . if > then we will delete the box
        prob_thresh : a threshold for proboblity of an object being captured bo the box. if < then we will drop the box
    """

    boxes=sorted(boxes,key=lambda x : x[0],reverse=True) #sort the boxes based on their object_prob
    boxes=np.array(boxes)  
    boxes=boxes[boxes[:,0]>prob_thresh] # drop the ones that have obejct_prob < prob_thresh

    trimed_boxes=[]
    while boxes.size >0:
        refrence=boxes[0]
        boxes=boxes[1:]
        iou_scores=calculate_iou(predictions=boxes[...,2:].tolist(),
                                 main_box=refrence[2:].tolist())

        indx=np.where(((iou_scores < iou_thresh)|(boxes[...,1:2]!=refrence[1])))[0].reshape(-1)
        trimed_boxes.append(refrence.tolist())
        boxes=boxes[indx]
    return trimed_boxes





