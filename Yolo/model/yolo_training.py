import torch
import torch.nn as nn
import numpy as np
import math

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(YOLOLoss,self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5+num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.ignore_threshold = 0.5
        self.cuda = cuda

    def forward(self, l,input,target):
        #----------------------------------------------------#
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets代表的是真实框。
        #----------------------------------------------------#
        #--------------------------------#
        #   获得图片数量，特征层的高和宽
        #   13和13
        #--------------------------------#

        batch_size = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #   stride_h和stride_w都是32。
        #-----------------------------------------------------------------------#
        stride_h = self.input_shape[0]/in_h
        stride_w = self.input_shape[1]/in_w

        scaled_anchors = [ (a_w/stride_w, a_h/stride_h) for a_w,a_h in self.anchors]
        scaled_anchors = torch.tensor(scaled_anchors)
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3*(5+num_classes), 13, 13 => batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #-----------------------------------------------#
        prediction = input.view(batch_size, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w ).permute(0,1,3,4,2).contiguous()

        # 先验框中心的参数调整

        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])

        # 先验框的宽高调整参数
        w = prediction[...,2]
        h = prediction[...,3]

        # 获取置信度，是否有物体
        conf = torch.sigmoid(prediction[...,4])

        # 种类置信度
        pred_cls = torch.sigmoid(prediction[...,5:])

        # 获取网络目标预测结果

        y_true, noobj_mask,box_loss_scale = self.get_target(l,target,scaled_anchors,in_h,in_w)

        # 将预测结果进行解码，判断预测结果和真实值的重合度
        # 如果重合程度大，则忽略， 因为这些特征点属于预测比较准确的特征点，作为负样本不合适
        noobj_mask = self.get_ignore(l,x,y,h,w,target,scaled_anchors,in_h,in_w,noobj_mask)

        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()

        # 目前不清楚该部分具体内容
        box_loss_scale = 2-box_loss_scale

        # 计算中心偏移情况的loss
        loss_x = torch.sum(self.BCELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4])
        loss_y = torch.sum(self.BCELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])
        # 计算宽高调整值的loss
        loss_w = torch.sum(self.MSELoss(w, y_true[..., 2]) * 0.5 * box_loss_scale * y_true[..., 4])
        loss_h = torch.sum(self.MSELoss(h, y_true[..., 3]) * 0.5 * box_loss_scale * y_true[..., 4])
        #   计算置信度的loss
        loss_conf   = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                      torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask)

        loss_cls    = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))

        loss        = loss_x  + loss_y + loss_w + loss_h + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos

    def get_target(self,l,target,anchors,in_h,in_w):
        batch_size = len(target)

        # 获取哪些框不包含物体，requires_grad=False用于说明当前量在计算中是否需要保留梯度信息
        noobj_mask = torch.ones(batch_size, len(self.anchors_mask[l]), in_h,in_w,requires_grad=False)
        # 让网络更加关注小目标
        box_loss_scale = torch.zeros(batch_size, len(self.anchors_mask[l]), in_h,in_w,requires_grad=False )

        y_true = torch.zeros(batch_size, len(self.anchors_mask[l]), in_h,in_w,self.bbox_attrs,requires_grad=False)

        for b in range(batch_size):
            if len(target[b])==0:
                continue

            batch_target = torch.zeros_like(target[b])

            # 计算正样本在特征层的中心点
            batch_target[:,[0,2]] = target[b][:,[0,2]]*in_w
            batch_target[:,[1,3]] = target[b][:,[1,3]]*in_h
            batch_target[:,4] = target[b][:,4]

            batch_target = batch_target.cpu()

            # 将真实框的表达方式转换 num_true_box, 4,其中0,1是0，2，3是经过计算的宽高
            gt_box = torch.cat((torch.zeros((batch_target.size(0),2)), batch_target[:,2:4]),1)
            gt_box = torch.FloatTensor(gt_box)

            # 将先验框的表达方式转换
            anchor_shape = torch.cat((torch.zeros((len(anchors),2)), anchors),1)
            anchor_shape = torch.FloatTensor(anchor_shape)

            # 计算交兵比
            # self.calculate_iou(gt_box, anchor_shapes) = [num_true_box,9]每一个真实框和9个先验框的重合情况
            # best_ns
            # 每个真实框的重合度max_iou，每一个真实框最重合的先验框的序号  重合度=两个框的重合部分/两个框的总面积
            best_ns = torch.argmax(self.calculate_iou(gt_box,anchor_shape),dim = -1)

            for t, best_n in enumerate(best_ns):
                # 如果重合率最大的真实框不属于当前维度，跳过
                if best_n not in self.anchors_mask[l]:
                    continue
                k = self.anchors_mask[l].index(best_n)

                # 获取真实框属于哪个网格点
                i = torch.floor(batch_target[t,0]).long()
                j = torch.floor(batch_target[t,1]).long()

                # 取出真实框的种类
                c = batch_target[t,4].long()
                noobj_mask[b,k,j,i] = 0

                y_true[b,k,j,i,0] = batch_target[t,0]-i.float()
                y_true[b,k,j,i,1] = batch_target[t,1]-i.float()
                y_true[b,k,j,i,2] = math.log(batch_target[t,2]/anchors[best_n][0])
                y_true[b,k,j,i,3] = math.log(batch_target[t,3]/anchors[best_n][1])
                y_true[b,k,j,i,4] = 1
                y_true[b,k,j,i,c+5] = 1

                #----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                #----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale
    def calculate_iou(self,_box_a,_box_b):
        # 计算真实框的左上角和右下角
        b1_x1 = _box_a[:, 0] - _box_a[:, 2] / 2
        b1_x2 = _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1 = _box_a[:, 1] - _box_a[:, 3] / 2
        b1_y2 = _box_a[:, 1] + _box_a[:, 3] / 2

        # 计算先验框获得的预测框的左上角和右下角
        b2_x1 = _box_b[:, 0] - _box_b[:, 2] / 2
        b2_x2 = _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1 = _box_b[:, 1] - _box_b[:, 3] / 2
        b2_y2 = _box_b[:, 1] + _box_b[:, 3] / 2

        # 将真实框和预测框都转换为左上角和右下角
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:,0], box_a[:,1], box_a[:,2], box_a[:,3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:,0], box_b[:,1], box_b[:,2], box_b[:,3] = b2_x1, b2_y1, b2_x2, b2_y2
        
        # A=真实框的数量 B = 先验框的数量
        A = box_a.size(0)        
        B = box_b.size(0)

        # 计算交的面积
        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1]

        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

        union = area_a + area_b - inter
        
        return inter / union  # [A,B]
    def get_ignore(self,l,x,y,h,w,target,scaled_anchors, in_h,in_w,noobj_mask):

        batch_size = len(target)
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心的左上角
        grid_x = torch.linspace(0,in_w-1,in_w).repeat(in_h,1).repeat( int( batch_size * len(self.anchors_mask[l])),1,1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0,in_h-1,in_h).repeat(in_w,1).t().repeat( int(batch_size*len(self.anchors_mask[l])) ,1,1 ).view(y.shape).type(FloatTensor)
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))
        
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后先验框的中心和宽高
        pred_boxes_x = torch.unsqueeze(x.data + grid_x,-1)
        pred_boxes_y = torch.unsqueeze(y.data + grid_y,-1)

        pred_boxes_w = torch.unsqueeze(torch.exp(w.data) * anchor_w,-1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h.data) * anchor_h,-1)

        pred_boxes = torch.cat([pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h],-1)

        for b in range(batch_size):
            pred_boxes_for_ignore = pred_boxes[b].view(-1,4)

            # 计算出正样本的特征层的中心点
            if len(target[b])>0:
                batch_target = torch.zeros_like(target[b])
                batch_target[:, [0,2]] = target[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = target[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4]

                #-------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)
    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
if __name__=="__main__":
    anchors = [
    [116,90],[156,198],[373,326],# 13x13
    [30,61], [62,45],[59,119],# 26x26
    [10,13],[16,30],[33,23]# 52x52
    ]

    loss = YOLOLoss(anchors,20,(416,416),False)
    input = torch.randn(16, 75,13,13)
    target = torch.randn(16,13,5)
    x = loss(1,input,target)



