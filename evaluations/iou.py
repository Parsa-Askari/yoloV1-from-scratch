import torch
def calculate_iou(predictions,main_box,form="circle"):
    """
    predictions shape (batch_size,4) [x1 , y1 ,x2 ,y2] for triangle and [x1 , y1 ,h1 ,w1] for circle
    main box shape (batch_size , 4) [x1 , y1 ,x2 ,y2] for triangle and [x1 , y1 ,h1 ,w1] for circle
    """
    if(form=="triangle"):
        x1_p=predictions[:,0:1]
        y1_p=predictions[:,1:2]
        x2_p=predictions[:,2:3]
        y2_p=predictions[:,3:4]

        x1_m=main_box[:,0:1]
        y1_m=main_box[:,1:2]
        x2_m=main_box[:,2:3]
        y2_m=main_box[:,3:4]
    else:
        x1_p=predictions[:,0:1] - predictions[:,2:3] / 2
        y1_p=predictions[:,1:2] - predictions[:,3:4] / 2
        x2_p=predictions[:,0:1] + predictions[:,2:3] / 2
        y2_p=predictions[:,1:2] + predictions[:,3:4] / 2

        x1_m=main_box[:,0:1] - main_box[:,2:3] / 2
        y1_m=main_box[:,1:2] - main_box[:,3:4] / 2
        x2_m=main_box[:,0:1] + main_box[:,2:3] / 2
        y2_m=main_box[:,1:2] + main_box[:,3:4] / 2
    # calculate intersection
    # (x1_i,y1_i)=max (x1_p , x1_m),max (y1_p , y1_m)
    # (x2_i,y2_i)=min (x2_p , x2_m),min (y2_p , y2_m)
    x1_i=torch.max(x1_p,x1_m)
    y1_i=torch.max(y1_p,y1_m)
    x2_i=torch.min(x2_p,x2_m)
    y2_i=torch.min(y2_p,y2_m)

    intersection = torch.clamp((x1_i-x2_i),min=0)*torch.clamp((y1_i-y2_i),min=0)

    union = (abs(x1_p - x2_p) * abs(y1_p - y2_p)) + (abs(x1_m - x2_m) * abs(y1_m - y2_m)) - intersection
    return intersection / (union + 1e-6)



