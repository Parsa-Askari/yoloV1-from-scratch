from .iou import calculate_iou
import torch
def avrage_precision_score(pred_boxes,target_boxes,iou_thresh=0.5,C=20):
    """
    pred_boxes : [img_idx , best_prob , best_class , x , y , w , h]
    """
    TP=torch.zeros(len(pred_boxes))
    classes_detected=torch.zeros(len(pred_boxes))
    classes_seen=set()
    for i,pbox in enumerate(pred_boxes):
        img_id = pbox[0]
        best_class = pbox[2]
        true_objects=target_boxes[img_id].get(best_class,[])
        for obj in true_objects:
            iou_score=calculate_iou(predictions=pbox[3:],main_box=obj[2:])
            if(iou_score>=0.5):
                TP[i]=1
                if(best_class not in classes_seen):
                    classes_seen.add(best_class)
                    classes_detected[i]=1
                break
    
    precision=(torch.cumsum(TP,dim=-1))/torch.arange(1,len(pred_boxes)+1)
    recall=(torch.cumsum(classes_detected,dim=-1))/C
    return torch.trapezoid(precision,recall)
