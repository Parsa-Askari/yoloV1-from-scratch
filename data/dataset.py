import PIL.Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import pandas as pd
import cv2 as cv
import PIL
class dataset(Dataset):
    def __init__(self,csv_path,S=7,B=2,C=20):
        super(dataset,self).__init__()
        df=pd.read_csv(csv_path)
        self.img_paths=df["img_paths"].tolist()
        self.bounding_paths=df["label_path"].tolist()
        self.B=B
        self.C=C
        self.S=S
        self.transform=Compose([transforms.Resize((448,448)),transforms.ToTensor()])
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, index):
        img_path=self.img_paths[index]
        bounding_path=self.bounding_paths[index]
        grid=torch.zeros(self.S,self.S,self.B*5+self.C)

        with open(bounding_path,"r") as f:
            boxes=f.readlines()
        for i in range(len(boxes)):
            texts=boxes[i].replace("\n","").split()
            new_box=[float(x) for x in texts[1:]]
            new_box=[int(texts[0])]+new_box
            boxes[i]=new_box

        img=PIL.Image.open(img_path)
        img=self.transform(img)
        for box in boxes:
            c,x,y,w,h,W,H=box
            i,x=int(x/(W/self.S)),(x/(W/self.S))-int(x/(W/self.S))
            j,y=int(y/(H/self.S)),(y/(H/self.S))-int(y/(H/self.S))
            w=self.S*w
            h=self.S*h
            if(grid[i,j,20]==0):
                grid[i,j,21:25]=torch.tensor([x,y,w,h])
                grid[i,j,20]=1
                grid[i,j,c]=1
        return img,boxes
