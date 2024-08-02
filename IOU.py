import torch
def iou(box1,box2,to_torch=False):
    """
        [w_c,h_c,w,h]
    """
    if(to_torch):
        box1=torch.from_numpy(box1)
        box2=torch.from_numpy(box2)
    x1_box1=box1[...,0:1]-(box1[...,2:3]/2)
    y1_box1=box1[...,1:2]-(box1[...,3:]/2)
    x2_box1=box1[...,0:1]+(box1[...,2:3]/2)
    y2_box1=box1[...,1:2]+(box1[...,3:]/2)

    x1_box2=box2[...,0:1]-(box2[...,2:3]/2)
    y1_box2=box2[...,1:2]-(box2[...,3:]/2)
    x2_box2=box2[...,0:1]+(box2[...,2:3]/2)
    y2_box2=box2[...,1:2]+(box2[...,3:]/2)

    xi_2=torch.min(x2_box1,x2_box2)
    xi_1=torch.max(x1_box1,x1_box2)
    yi_1=torch.max(y1_box1,y1_box2) 
    yi_2=torch.min(y2_box1,y2_box2)

    intersection=torch.clamp(xi_2-xi_1,min=0)*torch.clamp(yi_2-yi_1,min=0)
    union=torch.clamp(x2_box1-x1_box1,min=0)*torch.clamp(y2_box1-y1_box1,min=0) \
         + torch.clamp(x2_box2-x1_box2,min=0)*torch.clamp(y2_box2-y1_box2,min=0) \
         - intersection
    
    return (intersection / (union + 1e-6))