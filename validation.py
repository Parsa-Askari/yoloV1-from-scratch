from helpers import *
from data.dataset import *
from MAP import *
from NMX import nmx
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
@torch.no_grad()
def validate(dataloader,model,device,prob_thresh=0.5,iou_thresh=0.6,C=20):
    model.eval()
    final_predictions={i:{} for i in range(C)}
    final_targets={i:{} for i in range(C)}
    pred_classes={i:0 for i in range(C)}
    target_classes={i:0 for i in range(C)}
    img_index=0
    for img , grid in tqdm(dataloader):
        target_grids=grid.reshape(-1,49,30)
        grids=model(img.to(device)).reshape(-1,49,30)
        target_grids=get_best_boxes(target_grids)
        grids=get_best_boxes(grids.cpu())
        target_grids[...,1:]=reverse_box(target_grids[...,1:])
        grids[...,1:]=reverse_box(grids[...,1:])
        batch_size=grids.shape[0]
        for i in range(batch_size):
            boxes=grids[i].tolist()
            target_boxes=target_grids[i].tolist()
            # print(target_boxes.shape)
            # print(boxes.shape)
            boxes=nmx(boxes,iou_thresh,prob_thresh)
            for box in boxes : 
                if(img_index not in final_predictions[box[0]]):
                    final_predictions[box[0]][img_index]=[]
#                 print(boxes)
                final_predictions[box[0]][img_index].append(box)
                pred_classes[box[0]]+=1
            for box in target_boxes : 
                if(box[1]<prob_thresh):
                    continue
                
                if(img_index not in final_targets[box[0]]):
                    final_targets[box[0]][img_index]=[]
                final_targets[box[0]][img_index].append(box)
                target_classes[box[0]]+=1
            img_index+=1
        
    ap=maximum_avrage_precision(final_predictions,
                                final_targets,
                                pred_classes,
                                target_classes,
                                iou_thresh=iou_thresh)
    
    model.train()
    return ap
