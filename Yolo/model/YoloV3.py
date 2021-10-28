from typing import OrderedDict
import torch
import torch.nn as nn
from collections import OrderedDict
from model.mydarknet import DarkNet53

def conv2d(filter_in, filter_out, kernel_size):
    # 分情况，kernal_size == 1 时，pad = 0
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        OrderedDict([
            ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(filter_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ])
    )

def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        # ------------------------- #
        # 骨架  out_filters = [64, 128, 256, 512, 1024]
        # ------------------------- #
        self.backbone = DarkNet53()

        out_filters = self.backbone.layers_out_filters

        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1],len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv = conv2d(512,256,1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode = 'nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) *(num_classes + 5))

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128,len(anchors_mask[2]) * (num_classes + 5))
    
    def forward(self, x):
        x = self.backbone(x)
        x2 = x[0]
        x1 = x[1]
        x0 = x[2]
        # 此处不让last_layer0全部执行完毕，是因为需要提取中间值，进行上采样用。
        # 进行out0计算
        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)
        
        # 进行上采样和拼接
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in,x1],1)
        
        # 进行out1计算
        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)
        
        # 进行上采样和拼接
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in,x2],1)
        
        # 进行out2计算
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2
if __name__=="__main__":    
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
    model = YoloBody(anchors_mask,20)

    x = torch.randn(1,3,416,416)
    y = model(x)
    print(y[0].shape)
