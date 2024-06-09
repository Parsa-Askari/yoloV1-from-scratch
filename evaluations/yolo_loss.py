import torch
import torch.nn as nn
from .iou import calculate_iou
class Loss(nn.Module):
    def __init__(self,S,B,C):
        super(Loss,self).__init__()
        self.S=S
        self.B=B
        self.C=C
        self.lamda_coord=5
        self.lamda_noobj=0.5
    def forward(self,boxes,target_boxes):
        """
            boxes : shape (batch size , S*S*(B*5+C))
                [...,:20] : prob for each class
                [...,20] : obj is in box one
                [...,21:25] : x,y,w,h for box1
                [...,25] : obj is in box two
                [...,26:] x,y,w,h for box two

            target_boxes : shape (batch size , S*S*(B*5+C))
                [...,:20] : prob for each class
                [...,20] : obj is in box 
                [...,21:25] : x,y,w,h for box
        """
        boxes=boxes.reshape(-1,self.S,self.S,self.B*5+self.C)
        bests=self.find_best_box(boxes,target_boxes)
        has_object=target_boxes[...,20].unsqueeze(-1) # (batch size , S , S , 1)
        
        cord_loss=torch.mean(has_object*(
            ((bests)*boxes[...,26:28] 
            + (1-bests)*boxes[...,21:23]    
            - target_boxes[...,21:23])**2
        ))
        
        cord_loss+=torch.mean(has_object*(
            ((bests)*torch.sqrt(torch.abs(boxes[...,28:])) 
            + (1-bests)*torch.sqrt(torch.abs(boxes[...,23:25]))    
            - torch.sqrt(torch.abs(target_boxes[...,23:25]))
            )**2
        ))


        obj_loss=torch.mean(has_object*(
            ((bests)*boxes[...,:25:26]
            + (1-bests)*boxes[...,20:21]
            - target_boxes[...,20:21])**2
        ))

        noobj_loss=torch.mean((1-has_object)*(
            ((bests)*boxes[...,:25:26]
            + (1-bests)*boxes[...,20:21]
            - target_boxes[...,20:21])**2
        ))

        class_loss=torch.mean(has_object*(
            (boxes[...,:20]
            - target_boxes[...,:20])**2
        ))

        total_loss=self.lamda_coord*cord_loss 
        + obj_loss 
        + self.lamda_noobj*noobj_loss
        + class_loss
        
        return total_loss
    
    def find_best_box(self,boxes,target_boxes):
        iou1=calculate_iou(predictions=boxes[...,21:25],
                           main_box=target_boxes[...,21:25]).unsqueeze(0)
        iou2=calculate_iou(predictions=boxes[...,26:],
                           main_box=target_boxes[...,21:25]).unsqueeze(0)
        bests=torch.argmax(torch.cat([iou1,iou2],dim=0),dim=0)
        return bests # shape : (batch size, S, S, 1)
        