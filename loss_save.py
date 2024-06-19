import torch
import torch.nn as nn
# from .iou import calculate_iou
class Loss(nn.Module):
    def __init__(self,S,B,C):
        super(Loss,self).__init__()
        self.S=S
        self.B=B
        self.C=C
        self.lamda_coord=5
        self.lamda_noobj=0.5
        self.mse=nn.MSELoss(reduction="sum")
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
        best_box=has_object*((bests)*boxes[...,25:] + (1-bests)*boxes[...,20:25])
        noobj_box1=(1-has_object)*(boxes[...,20:21])
        noobj_box2=(1-has_object)*(boxes[...,25:26])
        """
        cordination losses
        """
        
        cord_loss=torch.sum((
                best_box[...,1:3].reshape(-1,2)
                -(has_object*target_boxes[...,21:23]).reshape(-1,2)
        )**2)
        
        cord_loss+=torch.sum((
             (torch.sign(best_box[...,3:5])*torch.sqrt(torch.abs(best_box[...,3:5])+1e-6)).reshape(-1,2)
            -(has_object*torch.sign(target_boxes[...,23:25])*torch.sqrt(torch.abs(target_boxes[...,23:25])+1e-6)).reshape(-1,2)
        )**2)
        
        
#         print(cord_loss)

        """
        obj losses
        """
        obj_loss=torch.sum((
            best_box[...,0:1].reshape(-1)
            - (has_object*target_boxes[...,20:21]).reshape(-1)
        )**2)

        
#         print(obj_loss)
        """
        noobj losses
        """
        noobj_loss=torch.sum((
            noobj_box1.reshape(-1,49)
            - ((1-has_object)*target_boxes[...,20:21]).reshape(-1,49)
        )**2)

#         print(noobj_loss)
        noobj_loss+=torch.sum((
            noobj_box2.reshape(-1,49)
            - ((1-has_object)*target_boxes[...,20:21]).reshape(-1,49)
        )**2)
      
       

        """
        class loss
        """
        class_loss=torch.sum((
            (has_object*boxes[...,:20]).reshape(-1,20)
            -(has_object*target_boxes[...,:20]).reshape(-1,20)
        )**2)
#         print(class_loss)

        total_loss=(self.lamda_coord*cord_loss 
        + obj_loss 
        + self.lamda_noobj*noobj_loss
        + class_loss)
        
#         print(total_loss)
        return total_loss
    
    def find_best_box(self,boxes,target_boxes):
        iou1=calculate_iou(predictions=boxes[...,21:25],
                           main_box=target_boxes[...,21:25]).unsqueeze(0)
        iou2=calculate_iou(predictions=boxes[...,26:],
                           main_box=target_boxes[...,21:25]).unsqueeze(0)
        bests=torch.argmax(torch.cat([iou1,iou2],dim=0),dim=0)
        return bests # shape : (batch size, S, S, 1)
        