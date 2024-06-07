from data import dataset
from model import yolo
from torch.utils.data import DataLoader

ds=dataset("../VOCdevkit/train.csv")
dataloader=DataLoader(ds,batch_size=4)
yolo_model=yolo(S=7,B=2,C=20)
for img , grid in dataloader:
    print(img.shape)
    out=yolo_model(img)
    print(out.shape)
    break