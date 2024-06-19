import cv2 as cv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from data import dataset
import numpy as np
def draw(boxes,img):
    """
        boxes : [S,S,B*5+C]
    """
    numpy_image = img.numpy()
    numpy_image = (numpy_image * 255).astype(np.uint8)
    numpy_image = np.transpose(numpy_image, (1, 2, 0))
    opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)

    box1=boxes[...,20:25]
    box2=boxes[...,25:]
    concated_boxes=torch.cat((boxes[...,20:21].unsqueeze(0),
                              boxes[...,25:26].unsqueeze(0)),dim=0)
    best_probs,best_box_idx=torch.max(concated_boxes,dim=0)

    
    best_box=best_box_idx*box2 + (1-best_box_idx)*box1  
    best_box=best_box.reshape(-1,5)
    classes=boxes[...,:20].reshape(-1,20)
    for i,box in enumerate(best_box):
        if(box[0]!=0):
            class_list=classes[i]
            print(box[1],box[3])
            print(box[2],box[4])
            x1,y1=int((box[1]-box[3]/2)*img.shape[2]),int((box[2]-box[4]/2)*img.shape[1])
            x2,y2=int((box[1]+box[3]/2)*img.shape[2]),int((box[2]+box[4]/2)*img.shape[1])
            print(x1,y1)
            print(x2,y2)
            
            opencv_image=cv.rectangle(opencv_image,(x1,y1),(x2,y2),color=(0, 255, 0) )
            # cv.imshow("s",opencv_image)

    # cv.waitKey(0)

train_ds=dataset("../VOCdevkit/train.csv")
Trainloader=DataLoader(train_ds,batch_size=1,pin_memory=False,num_workers=2)

for img,boxes in Trainloader:
    boxes=boxes.squeeze(0)
    img=img.squeeze(0)
    print(img.shape)
    draw(boxes,img)
    break