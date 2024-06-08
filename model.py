import torch
import torch.nn as nn
from modelArch import architecture
class conv(nn.Module):
    def __init__(self,inp,out,kernel_size,stride,padd):
        super(conv,self).__init__()
        self.conv=nn.Conv2d(in_channels=inp,
                            out_channels=out,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padd)
        self.normilizer=nn.BatchNorm2d(num_features=out)
        self.relu=nn.LeakyReLU()
    def forward(self,x):
        y=self.conv(x)
        y=self.normilizer(y)
        y=self.relu(y)
        return y
class ConvNetwork(nn.Module):
    def __init__(self,architecture,inp):
        super(ConvNetwork,self).__init__()
        self.layers=self.create(architecture,inp)
    def create(self,architecture,inp):
        layers=[]
        for Layer in architecture:
            if(type(Layer)==list):
                for L in Layer:
                    if(type(L)==list):
                        # print(L)
                        for i in range(L[2]):
                            layers.append(conv(inp=inp,
                                               out=L[0][1],
                                               kernel_size=L[0][0],
                                               stride=L[0][2],
                                               padd=L[0][3]))
                            inp=L[0][1]
                            layers.append(conv(inp=inp,
                                               out=L[1][1],
                                               kernel_size=L[1][0],
                                               stride=L[1][2],
                                               padd=L[1][3]))
                            inp=L[1][1]
                    else:
                        layers.append(conv(inp=inp,
                                            out=L[1],
                                            kernel_size=L[0],
                                            stride=L[2],
                                            padd=L[3]))
                        inp=L[1]
            else:
                layers.append(nn.MaxPool2d(kernel_size=(Layer[0],Layer[0]),
                                           stride=(Layer[1],Layer[1])))
        return nn.ParameterList(layers)
    def forward(self,x):
        for L  in self.layers:
            x=L(x)
        return x
class FullyConnected(nn.Module):
    def __init__(self,S,B,C):
        super(FullyConnected,self).__init__()
        self.layers=nn.ParameterList([
            nn.Flatten(),
            nn.Linear(S*S*1024,4096),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(4096,S*S*(C+(B*5)))
        ])
    def forward(self,x):
        for L in self.layers:
            x=L(x)
        return x
class YOLO(nn.Module):
    def __init__(self,S,B,C):
        super(YOLO,self).__init__()
        self.conv=ConvNetwork(architecture,inp=3)
        self.connected=FullyConnected(S,B,C)
    def forward(self,x):
        x=self.conv(x)
        x=self.connected(x)
        return x