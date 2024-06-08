from data import dataset
from model import YOLO
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from evaluations import Loss

EPOCHS=1
BATCH_SIZE=4
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
    for ep in range(EPOCHS):
        for img , grid in Trainloader:
            out=yolo(img)
            loss=loss_fn(out,grid)
            break
Train()