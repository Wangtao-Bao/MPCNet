import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
torch.pi=math.pi


def FocalIoULoss(inputs, targets):
    "Non weighted version of Focal Loss"

    # def __init__(self, alpha=.25, gamma=2):
    #     super(WeightedFocalLoss, self).__init__()
    # targets =
    # inputs = torch.relu(inputs)
    [b, c, h, w] = inputs.size()

    inputs = torch.nn.Sigmoid()(inputs)
    inputs = 0.999 * (inputs - 0.5) + 0.5
    BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
    intersection = torch.mul(inputs, targets)
    smooth = 1

    IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)

    alpha = 0.75
    gamma = 2
    num_classes = 2
    # alpha_f = torch.tensor([alpha, 1 - alpha]).cuda()
    # alpha_f = torch.tensor([alpha, 1 - alpha])
    gamma = gamma
    size_average = True

    pt = torch.exp(-BCE_loss)

    F_loss = torch.mul(((1 - pt) ** gamma), BCE_loss)

    at = targets * alpha + (1 - targets) * (1 - alpha)

    F_loss = (1 - IoU) * (F_loss) ** (IoU * 0.5 + 0.5)

    F_loss_map = at * F_loss

    F_loss_sum = F_loss_map.sum()

    return F_loss_map, F_loss_sum


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            mp= nn.MaxPool2d(2, 2)
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                if i>5:
                    gt_masks=mp(gt_masks)
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def forward(self, pred_log, target, epoch, with_shape=True):


        pred = pred_log
        smooth = 0.0
        # 计算预测和目标的交集
        intersection = pred * target

        # 对交集、预测和目标张量的每个样本分别进行求和
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        # 计算预测和目标面积差异的平方
        dis = torch.pow((pred_sum - target_sum) / 2, 2)
        # 根据面积差异计算权重
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        # 在训练的早期阶段（epoch <= warm_epoch），仅使用 IoU 损失。之后，结合形状感知损失和加权的 IoU 损失
        if epoch > 5:
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


#用于计算两个张量 pred 和 target 之间的损失。它通过比较预测的和目标的特征中心位置及其方向来衡量二者之间的差异。
def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth))
                                                           - torch.arctan(
            (target_centery) / (target_centerx + smooth))))

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            mp = nn.MaxPool2d(2, 2)
            for i in range(len(preds)):
                pred = preds[i]
                if i >5:
                    gt_masks = mp(gt_masks)
                # 确保形状一致
                if pred.size() != gt_masks.size():
                    gt_masks = gt_masks.repeat(pred.size(0), 1, 1, 1)  # 扩展批次大小

                pred =  pred.float()
                gt_masks = gt_masks.float()
                # print(pred.shape, gt_masks.shape)
                # print(pred.min().item(), pred.max().item())
                # print(gt_masks.min().item(), gt_masks.max().item())
                loss = self.bce_loss(pred, gt_masks)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            # 确保形状一致
            if pred.size() != gt_masks.size():
                gt_masks = gt_masks.repeat(pred.size(0), 1, 1, 1)  # 扩展批次大小
            loss = self.bce_loss(pred, gt_masks)
            return loss

class FindCoarseEdge(nn.Module):
    def __init__(self):
        super(FindCoarseEdge, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return torch.sigmoid(x0)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.soft_iou_loss = SoftIoULoss()
        self.edge_extractor = FindCoarseEdge()
        self.bce_loss = BCELoss()

    def forward(self, preds, gt_masks):
        # 计算 SoftIoU 损失
        iou_loss = self.soft_iou_loss(preds, gt_masks)

        # 提取预测和标签的边缘
        pred_edges = [self.edge_extractor(pred) for pred in preds] if isinstance(preds, list) or isinstance(preds, tuple) else self.edge_extractor(preds)
        gt_edges = self.edge_extractor(gt_masks)

        bce_loss = self.bce_loss(pred_edges, gt_edges)

        # 最终损失是 SoftIoU 和 BCE 损失的和
        total_loss = iou_loss * 0.5 + bce_loss
        return total_loss