from .iou import calculate_iou
import torch
import numpy as np
def avrage_precision_score(pred_boxes  ,target_boxes ,pred_class_count,
                           target_class_count,iou_thresh=0.5,C=20):
    """
    predbox={class_id:[img_idx:[box1 , box2 , ... ]]}
        box : [prob , class , x , y , w , h]
    """
    totol_AP=[]
    for c , target_imgs in target_boxes.items():
        preds_imgs=pred_boxes[c]
        if(len(preds_imgs)==0 or len(target_imgs)==0):
            continue 
        TP=torch.zeros(len(pred_class_count[c]))
        FP=torch.zeros(len(pred_class_count[c]))

        found_boxes={id:torch.zeros(len(boxes)) for id,boxes in preds_imgs.item()}
        index=0
        for id , boxes in preds_imgs:
            boxes=sorted(boxes,key=lambda x : x[0],reverse=True)
            target_boxes=target_imgs[id]
            max_iou=0
            best_target_index=0
            for box in boxes : 
                for i,target_box in enumerate(target_boxes):
                    iou_s=calculate_iou(box[2:],target_box[2:])
                    if(iou_s >max_iou):
                        max_iou=iou_s
                        best_target_index=i
                if(max_iou>iou_thresh and found_boxes[id][best_target_index]!=1):
                    TP[index]=1
                    found_boxes[id][best_target_index]=1
                else:
                    FP[index]=1
                index+=1
        TP_CS=torch.cumsum(TP)
        precision=torch.cat(torch.tensor([1]) ,(TP_CS)/(TP_CS+torch.cumsum(FP) + 1e-6))
        recall=torch.cat(torch.tensor([0]),(TP_CS)/(target_class_count[c] + 1e-6))
        totol_AP.append(torch.trapezoid(precision,recall).itme())
    return np.mean(totol_AP)