from data import dataset
from model import YOLO
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from evaluations import Loss
from evaluations import evaluation
from tqdm.auto import tqdm
EPOCHS=1
BATCH_SIZE=5
S=7
B=2
C=20
LR=0.001
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY= True if DEVICE=="cuda" else False

train_ds=dataset("../VOCdevkit/train.csv")
Trainloader=DataLoader(train_ds,batch_size=BATCH_SIZE,pin_memory=True,num_workers=4)

yolo=YOLO(S=S,B=B,C=C)
optimizer=Adam(params=yolo.parameters(),lr=LR)
loss_fn=Loss(S,B,C)
def Train():
    """
    img = batch of images with shape (batch size , channel size , width , hight)
    img = batch of images with shape (batch size , S , S , B*5 + C)
    """
    with tqdm(total=EPOCHS) as main_pbar:
        for ep in range(EPOCHS):
            yolo=yolo.train()
            with tqdm(total=len(Trainloader)) as pbar:
                for img , grid in Trainloader:
                    out=yolo(img)
                    loss=loss_fn(out,grid)
                pbar.set_postfix({"loss":loss.item()})
                pbar.update(1)
            AP=evaluation(Trainloader,yolo)
            main_pbar.set_postfix({"AP":AP})
            main_pbar.update(1)
            
        
Train()