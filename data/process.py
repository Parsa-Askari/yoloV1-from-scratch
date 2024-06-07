import os
import xml.etree.ElementTree as ET
import pandas as pd
folder_path="../VOCdevkit"
voc7_folder="VOC2007"
voc12_folder="VOC2012"
image_paths={"train":[],"val":[],"test":[]}
image_label_paths={"train":[],"val":[],"test":[]}
boarder_path=os.path.join(folder_path,"boarders")
all_classes=[ "aeroplane", "bicycle", "boat", "bus",
              "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant",
                "sofa", "tvmonitor", "bird", "cat", "cow", "dog", "horse", "sheep", "person"]
os.makedirs(os.path.join(boarder_path,"voc7"),exist_ok=True)
os.makedirs(os.path.join(boarder_path,"voc12"),exist_ok=True)

def get_border_box(voc_path,voc_boarder_path,part,id):
    tree = ET.parse(f"{voc_path}/Annotations/{id}.xml")
    root = tree.getroot()
    size = root.find('size')
    W = int(size.find('width').text)
    H = int(size.find('height').text)
    with open(os.path.join(voc_boarder_path,f"{id}.txt"),"w") as f:
        image_label_paths[part].append(os.path.join(voc_boarder_path,f"{id}.txt"))
        for object in root.iter("object"):
            index=all_classes.index(object.find("name").text)
            cords=object.find("bndbox")
            x1=float(cords.find("xmin").text)
            x2=float(cords.find("xmax").text)
            y1=float(cords.find("ymin").text)
            y2=float(cords.find("ymax").text)
            c1=(x1+x2)/2
            c2=(y1+y2)/2
            w=(x2-x1)/W
            h=(y2-y1)/H
            
            f.write(f"{index} {c1} {c2} {w} {h} {W} {H} \n")

# 2007 dataset : Train valid test
partitions=["train","test","val"]
voc7_path=os.path.join(folder_path,voc7_folder)
voc7_boarder_path=os.path.join(boarder_path,"voc7")
for part in partitions:
    with open(f"{voc7_path}/ImageSets/Main/{part}.txt","r") as f:
        images=f.readlines()

    for id in images :
        id=id.replace("\n","")
        image_paths[part].append(os.path.join(voc7_path,f'JPEGImages/{id}.jpg'))
        get_border_box(voc7_path,voc7_boarder_path,part,id)
        
# 2012 dataset : Train valid
partitions=["train","val"]
voc12_path=os.path.join(folder_path,voc12_folder)
voc12_boarder_path=os.path.join(boarder_path,"voc12")
for part in partitions:
    with open(f"{voc12_path}/ImageSets/Main/{part}.txt","r") as f:
        images=f.readlines()
    for id in images :
        id=id.replace("\n","")
        image_paths[part].append(os.path.join(voc12_path,f'JPEGImages/{id}.jpg'))
        get_border_box(voc12_path,voc12_boarder_path,part,id)
# add 
image_paths["train"]+=image_paths["val"][:4000]
image_label_paths["train"]+=image_label_paths["val"][:4000]
image_label_paths["val"]=image_label_paths["val"][4000:]
image_paths["val"]=image_paths["val"][4000:]

pd.DataFrame({"img_paths":image_paths["train"],"label_path":image_label_paths["train"]}).to_csv(f"{folder_path}/train.csv",index=False)
pd.DataFrame({"img_paths":image_paths["val"],"label_path":image_label_paths["val"]}).to_csv(f"{folder_path}/valid.csv",index=False)
pd.DataFrame({"img_paths":image_paths["test"],"label_path":image_label_paths["test"]}).to_csv(f"{folder_path}/test.csv",index=False)
