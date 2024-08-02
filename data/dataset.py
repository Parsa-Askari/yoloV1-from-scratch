import PIL.Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
import pandas as pd
import os
import torch
import PIL.Image as Image
reverse_transfrom=Compose([
    transforms.ToPILImage(),
])

class VocDataset(Dataset):
    def __init__(self,csv_data_path,boxes_path,voc_path,S=7,B=2,C=20):
        super(VocDataset,self).__init__()
        data=pd.read_csv(csv_data_path)
        self.img_paths=data["data"].to_list()
        self.ids=data["ids"].to_list()
        self.boxes_path=boxes_path
        self.voc_path=voc_path
        self.transfrom=Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor()
        ])
        self.S=S
        self.B=B
        self.C=C
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        """
            S * S * (B*5 + C)
            [0:20] => classes
            [20:21] => pc1
            [21:25] => wc1,hc1,wb1,hb1
            [25:26] => pc2
            [26:30] => wc2,hc2,wb2,hb2 
        """
        img_path=os.path.join(self.voc_path,self.img_paths[index])
        img=Image.open(img_path)
        img=self.transfrom(img)
        grid=torch.zeros(self.S,self.S,(self.B*5)+self.C)
        with open(os.path.join(self.boxes_path,f"{self.ids[index]}.txt")) as f:
            boxes=f.readlines()
        for box in boxes:
            box=list(map(float,box[:-1].split(" ")))
            (class_id,W_t,H_t,w_c,h_c,w,h)=box
            class_id=int(class_id)
            w_c*=(448/W_t)
            h_c*=(448/H_t)
            w*=(448/W_t)
            h*=(448/H_t)
            W_t=448
            H_t=448
            # print(w_c,h_c,w,h)
            w_b=W_t/self.S
            h_b=H_t/self.S
            j=int(w_c//w_b)
            i=int(h_c//h_b)
            
            if(grid[i,j,20]==0):
                
                w_c=(w_c-(j*w_b))/w_b
                h_c=(h_c-(i*h_b))/h_b
                w=w/w_b
                h=h/h_b
                grid[i,j,class_id]=1
                grid[i,j,20]=1
                grid[i,j,21]=w_c
                grid[i,j,22]=h_c
                grid[i,j,23]=w
                grid[i,j,24]=h
        return img , grid