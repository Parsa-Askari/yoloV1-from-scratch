from data.dataset import *
from helpers import *
from validation import validate
from model import createNet , Loss
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np
train_path="./data/processed/train.csv"
valid_path="./data/processed/valid.csv"
boxes_path="./data/processed/boxes"
voc_path="./data/"
train_ds=VocDataset(train_path,boxes_path,voc_path)
train_loader=DataLoader(train_ds,batch_size=32,pin_memory=True,num_workers=4)
valid_ds=VocDataset(valid_path,boxes_path,voc_path)
valid_loader=DataLoader(valid_ds,batch_size=32,pin_memory=True,num_workers=4)
epochs=30
device="cuda" if torch.cuda.is_available() else "cpu"
model=createNet().to(device)
loss_fn=Loss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
for ep in tqdm(range(epochs)):
    val_steps=0
    loss_list=[]
    with tqdm(total=len(train_loader)) as pbar:
        for img , grid in train_loader:
            optimizer.zero_grad()
            out=model(img.to(device))
            out=out.reshape(-1,7*7,30)
            loss=loss_fn(out.cpu(),grid.cpu())
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().item())
            pbar.update(1)
        print(f"mean loss : {np.mean(loss_list)}")
        if((ep+1)%10==0):
            valid_ap=validate(valid_loader,model,device)
            print(f"train maximum avrage precision : {valid_ap}")
            
            print("==============")
        
#             pbar.set_postfix({"mean loss":np.mean(loss_list)})


    