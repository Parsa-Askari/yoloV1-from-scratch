import torch
import torch.nn as nn
from IOU import iou
network=[["conv",[(7,64,2,3)],1],
         ["max"],
         ["conv",[(3,192,1,1)],1],
         ["max"],
         ["conv",[(1,128,1,0)],1],
         ["conv",[(3,256,1,1)],1],
         ["conv",[(1,256,1,0)],1],
         ["conv",[(3,512,1,1)],1],
         ["max"],
         ["conv",[(1,256,1,0),(3,512,1,1)],4],
         ["conv",[(1,512,1,0)],1],
         ["conv",[(3,1024,1,1)],1],
         ["max"],
         ["conv",[(1,512,1,0),(3,1024,1,1)],2],
         ["conv",[(3,1024,1,1)],1],
         ["conv",[(3,1024,2,1)],1],
         ["conv",[(3,1024,1,1)],2]]
class Conv(nn.Module):
    def __init__(self,in_c,out_c,k,s,p):
        super(Conv,self).__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_c,out_c,k,s,p),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )
    def forward(self,inp):
        return self.layers(inp)
class FullyConnected(nn.Module):
    def __init__(self,S,B,C):
        super(FullyConnected,self).__init__()
        self.layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(S*S*1024,4096),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(4096,S*S*(C+(B*5)))       
        )
    def forward(self,inp):
        return self.layers(inp)
def createNet(S=7,B=2,C=20):
    model=nn.Sequential()
    in_c=3
    conv_i=0
    for layer in network:
        if(layer[0]=="max"):
            model.add_module(f"max_{conv_i}",nn.MaxPool2d(2,2))
            conv_i+=1
        else:
            dims=layer[1]
            repeat=layer[2]
            for _ in range(repeat):
                for dim in dims:
                    k=dim[0]
                    out_c=dim[1]
                    s=dim[2]
                    p=dim[3]
                    model.add_module(f"conv_{conv_i}",Conv(in_c,out_c,k,s,p))
                    in_c=out_c
                    conv_i+=1
    model.add_module("fully",FullyConnected(S,B,C))
    return model


class Loss(nn.Module):
    def __init__(self,S=7,B=2,C=20,mode="sum"):
        super(Loss,self).__init__()
        self.S=S
        self.B=B
        self.C=C
        self.lamda_coord=5
        self.lamda_noobj=0.5
        self.mse=nn.MSELoss(reduction=mode)
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
        
        cord_loss=self.mse(
                torch.flatten(best_box[...,1:3],end_dim=-2),
                torch.flatten(has_object*target_boxes[...,21:23],end_dim=-2)
        )
        #torch.sign(best_box[...,3:5])
        #torch.sign(target_boxes[...,23:25])
        cord_loss+=self.mse(
             torch.flatten(torch.sign(best_box[...,3:5])*torch.sqrt(torch.abs(best_box[...,3:5]+1e-6)),end_dim=-2),
             torch.flatten(has_object*torch.sign(target_boxes[...,23:25])*torch.sqrt(torch.abs(target_boxes[...,23:25]+1e-6)),end_dim=-2)
        )
        
        
#         print(cord_loss)

        """
        obj losses
        """
        obj_loss=self.mse(
            torch.flatten(best_box[...,0:1]),
            torch.flatten(has_object*target_boxes[...,20:21])
        )

        
#         print(obj_loss)
        """
        noobj losses
        """
        noobj_loss=self.mse(
            torch.flatten(noobj_box1,start_dim=1),
            torch.flatten((1-has_object)*target_boxes[...,20:21],start_dim=1)
        )

#         print(noobj_loss)
        noobj_loss+=self.mse(
            torch.flatten(noobj_box2,start_dim=1),
            torch.flatten((1-has_object)*target_boxes[...,20:21],start_dim=1)
        )
      
       

        """
        class loss
        """
        class_loss=self.mse(
            torch.flatten(has_object*boxes[...,:20],end_dim=-2),
            torch.flatten(has_object*target_boxes[...,:20],end_dim=-2)
        )
#         print(class_loss)

        total_loss=(self.lamda_coord*cord_loss 
        + obj_loss 
        + self.lamda_noobj*noobj_loss
        + class_loss)
        
#         print(total_loss)
        return total_loss
    
    def find_best_box(self,boxes,target_boxes):
        iou1=iou(boxes[...,21:25],target_boxes[...,21:25]).unsqueeze(0)
        iou2=iou(boxes[...,26:],target_boxes[...,21:25]).unsqueeze(0)
        bests=torch.argmax(torch.cat([iou1,iou2],dim=0),dim=0)
        return bests # shape : (batch size, S, S, 1)