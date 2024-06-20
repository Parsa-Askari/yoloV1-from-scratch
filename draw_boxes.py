import cv2 as cv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from data import dataset
import numpy as np
def draw(boxes,img,W,H):
    """
        boxes : [S,S,B*5+C]
    """
    W=int(W)
    H=int(H)
    numpy_image = img.numpy()
    numpy_image = (numpy_image * 255).astype(np.uint8)
    numpy_image = np.transpose(numpy_image, (1, 2, 0))
    opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)
    opencv_image=cv.resize(opencv_image,(W,H))
    box1=boxes[...,20:25]
    box2=boxes[...,25:]
    concated_boxes=torch.cat((boxes[...,20:21].unsqueeze(0),
                              boxes[...,25:26].unsqueeze(0)),dim=0)
    best_probs,best_box_idx=torch.max(concated_boxes,dim=0)

    
    best_box=best_box_idx*box2 + (1-best_box_idx)*box1  
    best_box=best_box.reshape(-1,5)
    classes=boxes[...,:20].reshape(-1,20)
    for ind,box in enumerate(best_box):
        if(box[0]!=0):
            class_list=classes[ind]
            i,j=int(ind//7),int(ind%7)
            c1,c2=box[1].item(),box[2].item()
            c1=((c1+i)/7)*W
            c2=((c2+j)/7)*H
            print(c1,c2)
            w,h=box[3].item()/7,box[4].item()/7
            w=w*W
            h=h*H
            print(w,h)
            x1,y1=int(c1-w/2),int(c2-h/2)
            x2,y2=int(c1+w/2),int(c2+h/2)
            print(x1,y1)
            print(x2,y2)
            opencv_image=cv.rectangle(opencv_image,(x1,y1),(x2,y2),color=(0, 255, 0) )
            cv.imshow("s",opencv_image)

    cv.waitKey(0)

train_ds=dataset("../VOCdevkit/train.csv")
Trainloader=DataLoader(train_ds,batch_size=1,pin_memory=False,num_workers=2)

for img,boxes,W,H in Trainloader:
    boxes=boxes.squeeze(0)
    img=img.squeeze(0)
    draw(boxes,img,W.item(),H.item())
    break