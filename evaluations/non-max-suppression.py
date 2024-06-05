from iou import  calculate_iou
import numpy as np
def non_max_suppression(boxes,prob_thresh,iou_thresh):
    """
        boxes : bounding boxes . [object_prob,class,x,y,w,h]
        iou_thresh : a threshold for iou score . if > then we will delete the box
        prob_thresh : a threshold for proboblity of an object being captured bo the box. if < then we will drop the box
    """
    boxes=sorted(boxes,key=lambda x : x[0],reverse=True) #sort the boxes based on their object_prob
    boxes=np.array(boxes)  
    boxes=boxes[boxes[:,0]>=prob_thresh].tolist() # drop the ones that have obejct_prob < prob_thresh
    
    trimed_boxes=[]
    while boxes!=[]:
        refrence=boxes.pop(0)
        new_boxes=[box for box in boxes if calculate_iou(predictions=box[2:],main_box=refrence[2:]) <= iou_thresh or box[1]!=refrence[1]]
        trimed_boxes.append(refrence)
        boxes=new_boxes
    return trimed_boxes




