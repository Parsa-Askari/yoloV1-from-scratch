import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
base_folder="./VOCdevkit"
dataset_folders=["VOC2007","VOC2012"]
xml_folder="Annotations"
id_folder="ImageSets/Main"
img_folder="JPEGImages"
image_id_files={"VOC2007":["train","val","test"],
                "VOC2012":["train","val"]}
partitions={"train":[],"val":[],"test":[]}
partitions_ids={"train":[],"val":[],"test":[]}
boxes_path="./processed/boxes"
csv_paths="./processed/"
all_classes=[ 
            "aeroplane", "bicycle", "boat", "bus",
            "car", "motorbike", "train", "bottle", 
            "chair", "diningtable", "pottedplant",
            "sofa", "tvmonitor", "bird", "cat", "cow", 
            "dog", "horse", "sheep", "person"]

def makebox(obj,W_t,H_t):
    name=obj.find("name").text
    class_id=all_classes.index(name)
    bbox = obj.find('bndbox')
    xmin = float(bbox.find('xmin').text)
    ymin = float(bbox.find('ymin').text)
    xmax = float(bbox.find('xmax').text)
    ymax = float(bbox.find('ymax').text)
    w_c=(xmin+xmax)/2
    h_c=(ymin+ymax)/2
    h_b=ymax-ymin
    w_b=xmax-xmin
    
    return f"{class_id} {W_t} {H_t} {w_c} {h_c} {w_b} {h_b}"
def parse(xml_path,img_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size=root.find("size")
    W_t=int(size.find("width").text)
    H_t=int(size.find("height").text)
    objects=root.findall("object")
    boxes=[]
    for obj in objects:
        diff=obj.find("difficult").text
        if(diff=="1"):
            continue
        box=makebox(obj,W_t,H_t)
        boxes.append(box)
    
    content="\n".join(boxes)
    with open(os.path.join(boxes_path,f"{img_id}.txt"),"w") as f:
        f.write(content)
    
for dataset_folder in dataset_folders:
    folder=os.path.join(base_folder,dataset_folder)
    xml_path_base=os.path.join(folder,xml_folder)
    id_path_base=os.path.join(folder,id_folder)
    img_path_base=os.path.join(folder,img_folder)
    for part in image_id_files[dataset_folder]:
        with open(os.path.join(id_path_base,f"{part}.txt"),"r") as f:
            ids=f.readlines()
        for id in ids:
            id=id.split("\n")[0]
            xml_path=os.path.join(xml_path_base,f"{id}.xml")
            img_path=os.path.join(img_path_base,f"{id}.jpg")
            partitions[part].append(img_path[2:])
            partitions_ids[part].append(id)
            parse(xml_path,id)

valid_paths=np.array(partitions["val"][5000:])
train_paths=np.array(partitions["train"]+partitions["val"][:5000]+partitions["test"])

valid_paths_ids=np.array(partitions_ids["val"][5000:])
train_paths_ids=np.array(partitions_ids["train"] 
                        +partitions_ids["val"][:5000] 
                        +partitions_ids["test"])

train_csv=pd.DataFrame({"data":train_paths , "ids":train_paths_ids})
valid_csv=pd.DataFrame({"data":valid_paths , "ids":valid_paths_ids})
train_csv.to_csv(os.path.join(csv_paths,"train.csv"),index=False)
valid_csv.to_csv(os.path.join(csv_paths,"valid.csv"),index=False)
