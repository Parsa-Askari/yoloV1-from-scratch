import torch
from data.dataset import reverse_transfrom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def reverse_box(best_boxes,H_t=448,W_t=448):
    """
    S*S*10
    [0:1] => pc1
    [1:5] => wc1,hc1,wb1,hb1
    [5:6] => pc2
    [6:] => wc2,hc2,wb2,hb2 
    """
    # box1=boxes[...,:5]
    # box2=boxes[...,5:10]
    # best_boxes_idx=torch.argmax(torch.cat((box1[...,0:1].unsqueeze(0),box2[...,0:1].unsqueeze(0)),dim=0),dim=0)
    # best_boxes=(1-best_boxes_idx)*box1 + (best_boxes_idx)*box2
    best_boxes=best_boxes.reshape(-1,49,5)
    h_b=(H_t/7)
    w_b=(W_t/7)
    best_boxes[...,4:]=best_boxes[...,4:]*h_b
    best_boxes[...,3:4]=best_boxes[...,3:4]*w_b
    j_idx = torch.tensor(torch.arange(0,7)).repeat(7,1)
    i_idx=j_idx.T
    i_idx=i_idx.reshape(49,1)
    j_idx=j_idx.reshape(49,1)
    best_boxes[...,1:2]=(j_idx * w_b) + (best_boxes[...,1:2]*w_b)
    best_boxes[...,2:3]=(i_idx * h_b) + (best_boxes[...,2:3]*h_b)

    return best_boxes

def draw_boxes(img,boxes,prob=0.5):
    img=img.squeeze(0)
    boxes=boxes.squeeze(0)
    real_boxes=boxes[boxes[...,0]>prob]
    img=reverse_transfrom(img)
    fig, ax = plt.subplots()
    for box in real_boxes:
        x=int(box[1]-(box[3]/2))
        y=int(box[2]-(box[4]/2))
        w=int(box[3])
        h=int(box[4])
        ax.imshow(img)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
def get_best_boxes(boxes):
    """
        boxes = [B,S*S,30]
    """
    batch_size=boxes.shape[0]
    boxes=boxes.reshape(49*batch_size,30)
    class_ids=torch.argmax(boxes[...,:20],dim=1).unsqueeze(1)
    box1=boxes[...,20:25]
    box2=boxes[...,25:]
    best_box_ids=torch.argmax(torch.cat((box1[...,0:1].unsqueeze(0),
                    box2[...,0:1].unsqueeze(0)),dim=0),dim=0)
    best_boxes=(1-best_box_ids)*box1 + (best_box_ids*box2)
    best_boxes=torch.cat((class_ids,best_boxes),dim=-1)
    return best_boxes.reshape(batch_size,49,6)





# boxes=torch.tensor([[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.9,0,0,0,0,0.6,0,0,0,0],
#                     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7,0,0,0,0,0.8,0,0,0,0],
#                     [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0.2,0,0,0,0],
#                     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.9,0,0,0,0,0.2,0,0,0,0]])
# get_best_boxes(boxes)