import torch
from IOU import iou
import numpy as np
from tqdm.auto import tqdm
def maximum_avrage_precision(pred_boxes,target_boxes,pred_classes,target_classes,iou_thresh):
    """
        predbox={class_id:{img_idx:[box1 , box2 , ... ]}}
        target ={class_id:{img_idx:[box1 , box2 , ... ]}}
        box = [class_id,probs , w_c , h_c , w , h]
        pred_classes {class_id : number of acurance}
    """
    totol_AP=[]
    for c , target_images in target_boxes.items():
        pred_imgs=pred_boxes.get(c,{})
        if(len(pred_imgs)==0 or len(target_images)==0):
            continue 
        FP=torch.zeros(pred_classes[c])
        TP=torch.zeros(pred_classes[c])
        boxes_found={img_idx:torch.zeros(len(boxes)) for img_idx,boxes in target_images.items()}
        index=0
        for img_idx , boxes in pred_imgs.items():
            boxes=sorted(boxes , key= lambda x : x[1] , reverse=True)
            target_boxes=target_images.get(img_idx,[])
            for box in boxes:
                best_iou=0
                best_target_idx=0
                for i , target_box in enumerate(target_boxes):
#                     print(target_box[2:])
#                     print(box[2:])
                    iou_s=iou(torch.tensor(box[2:]),torch.tensor(target_box[2:]))
                    if(iou_s>best_iou):
                        best_iou=iou_s
                        best_target_idx=i
                if(best_iou>iou_thresh and boxes_found[img_idx][best_target_idx]!=1):
                    TP[index]=1
                    boxes_found[img_idx][best_target_idx]=1
                else:
                    FP[index]=1
                index+=1
        TP_CS=torch.cumsum(TP,dim=0)
        precision=torch.cat((torch.tensor([1]) ,(TP_CS)/(TP_CS+torch.cumsum(FP,dim=0) + 1e-6)),dim=0)
        recall=torch.cat((torch.tensor([0]),(TP_CS)/(target_classes[c] + 1e-6)),dim=0)
        totol_AP.append(torch.trapezoid(precision,recall).item())
    if(totol_AP==[]):
        return 0
    return np.mean(totol_AP)        