from pathlib import Path
import torch
from torch._C import Node
import torch.nn as nn
import numpy as np
from model.YoloV3 import *
from torchvision.ops import nms
anchors = [
    [116,90],[156,198],[373,326],# 13x13
    [30,61], [62,45],[59,119],# 26x26
    [10,13],[16,30],[33,23]# 52x52
    ]


class MyDecodeBox():
    def __init__(self,anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(MyDecodeBox,self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = num_classes+5
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
    def decode_box(self,inputs):
        outputs = []

        for i,input in enumerate(inputs):
            # 输入的input共有3个，分别是 
            # batch_size, 255, 13, 13 
            # batch_size, 255, 26, 26
            # batch_size, 255, 52, 52
            # 其中 255为3*85
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)
            # 求解步长
            stride_h = self.input_shape[0]/input_height
            stride_w = self.input_shape[1]/input_width
            # 
            scaled_anchors = [
                (anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]
                ]
            
            # 新的shape=[batch_size, 3, 13, 13, 25]
            prediction = input.view(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
            
            # 获取x y w h-4
            x = torch.sigmoid(prediction[..., 0])  
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[...,2]
            h = prediction[...,3]

            # 获取置信度-1
            conf = torch.sigmoid(prediction[...,4])

            # 获取种类
            pred_cls = torch.sigmoid(prediction[...,5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 计算偏移量
            # bx = sigmoid(tx)+Cx by = sigmoid(ty) + Cy
            # 其中 x = sigmoid(tx) y = sigmoid(y) Cx和Cy为左上角点，即偏移时，向右下角偏移。

            # 获取Cx和Cy
            grid_x = torch.linspace(0,input_width-1,input_height).repeat(input_height, 1).repeat(batch_size*len(self.anchors_mask[i]),1,1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0,input_height-1,input_width).repeat(input_width, 1).t().repeat(batch_size*len(self.anchors_mask[i]),1,1).view(x.shape).type(FloatTensor)

            #----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            # 获取张量的第2个维度，索引号为0的张量子集（第1列）
            # anchor_w.shape = 3*2
            # anchor_w[i][0] i in range(3)
            # 计算完成后，anchor_w.shape = 3*1
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            # 对每一个样本都要生成这样的宽高
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # 利用预测结果，对先验框进行调整
            pred_boxes = FloatTensor(prediction[...,:4].shape)
            # 计算偏移量
            pred_boxes[...,0] = x.data+grid_x
            pred_boxes[...,1] = y.data+grid_y
            # 计算实际的宽和高
            pred_boxes[...,2] = torch.exp(w.data)*anchor_w
            pred_boxes[...,3] = torch.exp(h.data)*anchor_h

            # 结果归一化
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            # out1的最后一个维度是4维，因此  可以通过广播 对_scale进行运算
            out1 = pred_boxes.view(batch_size, -1, 4) / _scale
            out2 = conf.view(batch_size,-1,1)
            out3 = pred_cls.view(batch_size,-1,self.num_classes)
            output = torch.cat((out1, out2, out3), -1)
            outputs.append(output)
        return outputs

    def yolo_correct_boxes(self, box_xy,box_wh, input_shape, image_shape, letterbox_image):

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # 此处offset为图像有效区域对于图像左上角的偏移情况
            # new_shape指宽高的缩放情况
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape)/2./input_shape
            scale = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        return boxes
    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        # prediction中。0，1，2，3分别表示，框的中心x，y，框的宽和高w,h 通过运算 将表示法换为  左上角坐标和右下角坐标表示法
        # priediction.shape = [batch_size,:,5+num_class]
        #  x1 = x-w/2 x2 = x+w/2
        # y1 = y-h/2 y2 = y+h/2
        box_corner = prediction.new(prediction.shape)
        box_corner[:,:,0] = prediction[:,:,0]-prediction[:,:,2]/2
        box_corner[:,:,2] = prediction[:,:,0]+prediction[:,:,2]/2
        box_corner[:,:,1] = prediction[:,:,1]-prediction[:,:,3]/2
        box_corner[:,:,3] = prediction[:,:,1]+prediction[:,:,3]/2

        # 将计算结果写回prediction
        prediction[:,:,:4] = box_corner[:,:,:4]

        output = [None for _ in range(len(prediction))]
        # prediction.shape = 16*10647*25
        for i, image_pred in enumerate(prediction):
            # 获取种类置信度和种类
            # 10647*25的第5列开始，一直向后，直到结束，选择最大的并返回索引。
            # 大小为10647*1
            class_conf, class_pred = torch.max(image_pred[:,5:5+num_classes],1,keepdim=True)
            # 利用置信度进行第一轮筛选
            # image_pred[:,4 ]存储的是，该区域是否有物体，
            conf_mask = (image_pred[:,4] * class_conf[:,0] >= conf_thres).squeeze()

            # 根据置信度，进行预测结果筛选
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue

            # 构造检测器
            # 检测器的内容：x1,y1,x2,y2,obj_conf, class_conf,class_pred
            # detection.shape = [num_anchors,7]
            detection = torch.cat( (image_pred[:,:5], class_conf.float(), class_pred.float()),1)
            # 选出detection中的不重复元素
            # detection的最后一列：class_pred，即 该框内包含的是哪个种类。
            unique_labels = detection[:,-1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()
            
            for c in unique_labels:
                # 在保留的结果中，寻找属于该类的结果，同一类的再进行抑制。
                detection_class = detection[detection[:,-1] == c]
                # 使用官方自带的非极大值抑制
                keep = nms(
                    detection_class[:,:4],
                    detection_class[:,4]*detection_class[:,5],
                    nms_thres
                    )
                max_detection = detection_class[keep]
                # 将结果进行拼接
                output[i] = max_detection if output[i] is None else torch.cat((output[i], max_detection))

            if output[i] is not None:
                output[i] = output[i].cpu().detach().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

            



if __name__=="__main__":
    anchors = torch.tensor([
    [10,13],[16,30],[33,23],# 52x52
    [30,61], [62,45],[59,119],# 26x26
    [116,90],[156,198],[373,326]# 13x13
    ])
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
    path = "Yolo\\modelresult.pth"

    myfile = Path(path)
    if myfile.is_file():
        y = torch.load(path,map_location="cpu")
    else:
        model = YoloBody(anchors_mask,20)
        model = model.to("cuda:0")
        x = torch.randn(16,3,416,416)
        x.to("cuda:0")
        y = model(x)
        torch.save(y,"modelresult.pth")
    
    decode = MyDecodeBox(anchors,20, torch.tensor([416,416]),anchors_mask)

    prediction = decode.decode_box(y)
    boxes = decode.non_max_suppression(torch.cat(prediction, 1),20,(416,416),(416,416),False)
    print(boxes)


