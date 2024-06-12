import torch
from .non_max_suppression import non_max_suppression
from .avrage_precision_score import avrage_precision_score
def split_boxes(boxes,S,B,C):
    """
        boxes : shpae (batch_size,S,S,B*5+C)
    """
    box1=boxes[...,21:25]
    box2=boxes[...,26:]
    best_prob,best=torch.max(torch.cat((boxes[...,20:21].unsqueeze(0),boxes[...,25:26].unsqueeze(0)),dim=0),dim=0)

    best_box=best*box2 + (1-best)*box1
    best_class=torch.argmax(boxes[...,:20],dim=-1).unsqueeze(-1)
    final_boxes=torch.cat((best_prob,best_class,best_box),dim=-1)
    
    return final_boxes.reshape(-1,S*S,6) # [prob , class , x , y , w ,h]
    
@torch.no_grad()
def evaluation(dataloader,model,prob_thresh=0.009,iou_thresh=0.5,S=7,B=2,C=20):
    model=model.eval()
    final_predictions={i:{} for i in range(C)}
    final_targets={i:{} for i in range(C)}
    pred_class_count={i:0 for i in range(C)}
    target_class_count={i:0 for i in range(C)}
    index=0
    for img,target in dataloader:
        pred=model(img)
        new_pred=split_boxes(pred.reshape(-1,S,S,B*5+C),S,B,C).tolist()
        new_target=split_boxes(target,S,B,C).tolist()
        batch_size=len(new_pred)
        for i in range(batch_size):
            cleaned_boxes=non_max_suppression(new_pred[i],prob_thresh,iou_thresh)
            for box in cleaned_boxes:
                if(index not in final_predictions[box[1]]):
                    final_predictions[box[1]][index]=[]
                final_predictions[box[1]][index].append(box)
                pred_class_count[box[1]]+=1
            for box in new_target[i]:
                if(box[0]>prob_thresh):
                    if(index not in final_targets[box[1]]):
                        final_targets[box[1]][index]=[]
                    final_targets[box[1]][index].append(box)
                    target_class_count[box[1]]+=1
            index+=1
        
        # print(len(final_predictions))
        
    AP=avrage_precision_score(final_predictions,
                              final_targets,
                              pred_class_count,
                              target_class_count,
                              iou_thresh,
                              C)
    return AP