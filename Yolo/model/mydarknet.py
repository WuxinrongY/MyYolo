from typing import OrderedDict
import torch.nn as nn
import torch
from torch.nn.modules.container import Sequential

class BasicBlock(nn.Module):
    def __init__(self, inplane, planes):
        super(BasicBlock,self).__init__()
        self.conv1_block = nn.Sequential(
            OrderedDict(
                [
                    ("Residual_conv1", nn.Conv2d(inplane,planes[0],1,1,0,bias=False)),
                    ("Residual_bn1",nn.BatchNorm2d(planes[0])),
                    ("Residual_relu1",nn.LeakyReLU(0.1))
                ]
            )
        )

        self.conv2_block = nn.Sequential(
            OrderedDict(
                [
                    ("Residual_conv2", nn.Conv2d(planes[0],planes[1],3,1,1,bias=False)),
                    ("Residual_bn2",nn.BatchNorm2d(planes[1])),
                    ("Residual_relu2",nn.LeakyReLU(0.1))
                ]
            )
        )

    def forward(self,x):
        residual = x

        out = self.conv1_block(x)
        out = self.conv2_block(out)

        out+=residual
        return out

class Darknet(nn.Module):
    def __init__(self, layers):
        super(Darknet,self).__init__()
        self.inplane = 32
        self.conv1_block = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3,32,3,1,1,bias=False)),
                    ("bn1",nn.BatchNorm2d(32)),
                    ("relu1",nn.LeakyReLU(0.1))
                ]
            )
        )
        self.layers_out_filters = [64, 128, 256, 512, 1024]

        self.Residual1 = self._make_layers([32,64],layers[0])
        self.Residual2 = self._make_layers([64,128],layers[1])
        self.Residual3 = self._make_layers([128,256],layers[2])
        self.Residual4 = self._make_layers([256,512],layers[3])
        self.Residual5 = self._make_layers([512,1024],layers[4])


    def forward(self,x):
        out = self.conv1_block(x)
        out = self.Residual1(out)
        out = self.Residual2(out)
        out3 = self.Residual3(out)
        out4 = self.Residual4(out3)
        out5 = self.Residual5(out4)

        res = [
            out3,
            out4,
            out5
        ]
        return res
    def _make_layers(self, planes,blocks):
        # 下采样
        layer = []
        layer.append(("conv1", nn.Conv2d(self.inplane,planes[1],3,2,1,bias=False)))
        layer.append(("bn1",nn.BatchNorm2d(planes[1])))
        layer.append(("relu1",nn.LeakyReLU(0.1)))
        
        self.inplane = planes[1]

        for i in range(blocks):
            layer.append(("Residual_{}".format(i), BasicBlock(self.inplane,planes)))
  
        return nn.Sequential(OrderedDict(layer))

def DarkNet53():
    model = Darknet([1,2,8,8,4])
    return model



if __name__=="__main__":
    x = torch.randn(1,3,416,416)
    model = Darknet([1,2,8,8,4])
    # print(model)
    T = model(x)
    print(T)



        