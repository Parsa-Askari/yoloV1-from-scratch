from data.dataset import *
from helpers import *
from validation import validate
from model import createNet , Loss
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
train_path="./data/processed/train.csv"
valid_path="./data/processed/valid.csv"
boxes_path="./data/processed/boxes"
voc_path="./data/"
train_ds=VocDataset(train_path,boxes_path,voc_path)
train_loader=DataLoader(train_ds,batch_size=32)
valid_ds=VocDataset(valid_path,boxes_path,voc_path)
valid_loader=DataLoader(valid_ds,batch_size=32)
epochs=5
device="cuda" if torch.cuda.is_available() else "cpu"
model=createNet().to(device)
loss_fn=Loss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
for ep in range(epochs):
    val_steps=0
    with tqdm(total=len(train_loader)) as pbar:
        for img , grid in train_loader:
            out=model(img.to(device))
            out=out.reshape(7,7,30)
            loss=loss_fn(out,grid)
            if((val_steps+1)%100==0):
                valid_ap=validate(valid_loader,model,device)
                print(valid_ap)
                print("==============")
            val_steps+=1
            pbar.set_postfix({"loss":loss})
            pbar.update(1)


    