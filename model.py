import torch
import numpy as np
from utils.box_tools import loc2box, create_anchor_all, generate_anchor_base,loc2box_torch
from torchvision.models import vgg16
from utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator, ROICreator, NMS
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from config import cfg
from torchvision.ops import nms, batched_nms, RoIPool


def decom_vgg16():
    # vgg16的网络结构 [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    # 这里下载预训练模型时,可以手动指定下载路径,具体 参考 vgg16 -> _vgg -> load_state_dict_from_url方法中model_dir参数
    model = vgg16()
    # 如果基于已有模型训练则不加载vgg模型,否则加载
    if not cfg.load_path:
        # 从Pytorch官方加载vgg的权重,model_dir为权重保存地址 '.'为当前目录下
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth', model_dir='.')
        model.load_state_dict(state_dict)
    # 截取vgg16的前30层网络结构,因为再往后的就不需要了
    # 第30层是 conv_5_3后面的 Relu, 31层为maxpool再往后就是fc层了
    features = list(model.features)[:30]
    classifier = model.classifier
    # Linear(in_features=25088, out_features=4096, bias=True)
    # ReLU(inplace=True)
    # Dropout(p=0.5, inplace=False)
    # Linear(in_features=4096, out_features=4096, bias=True)
    # ReLU(inplace=True)
    # Dropout(p=0.5, inplace=False)
    # Linear(in_features=4096, out_features=1000, bias=True)
    classifier = list(classifier)
    # 删除的是最后一层以及两个 dropout层
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)

    # 冻结前4层的卷积层 conv_relu_conv_relu_max *2
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    features = nn.Sequential(*features)
    # vgg的特征提取层 和最后的ROI_head分类部分
    return features, classifier


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.n_class = len(cfg.class_name)+1
        # 特征提取网络 RPN网络 ROI_Head网络 参数初始化时的均值与方差 以及rpn和roi的损失权重
        self.extractor, classifier = decom_vgg16()
        self.rpn = RegionProposalNetwork()
        self.head = RoIHead(n_class=self.n_class, classifier=classifier)
        self.nms_thresh = cfg.nms_roi  # 测试时 ROI阶段nms的IOU阈值
        self.rpn_sigma = cfg.rpn_sigma
        self.roi_sigma = cfg.roi_sigma
        # 为RPN及ROI_Head网络准备的 AnchorTargetCreator ProposalTargetCreator 优化方式
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.optimizer = self.get_optimizer()
        self.mean = torch.Tensor((0., 0., 0., 0.)).cuda().repeat(self.n_class)[None]
        self.std = torch.Tensor((0.1, 0.1, 0.2, 0.2)).cuda().repeat(self.n_class)[None]
        self.score_thresh = 0.05  # 训练及计算mAP时的ROI网络中的score阈值,实际推理时为0.7

    def forward(self, x, target_boxes=None, target_labels=None, scale=1.):
        img_size = x.shape[2:]
        features = self.extractor(x)
        # 这里把一个batch(虽然为1)中的所有roi都放在一起了,用roi_indices来代表其所属batch的index
        # [1, 16650, 4] [1, 16650, 2] [2000, 4] [2000,] [16650, 4]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)
        # 非训练阶段
        if not self.training:
            # (300, self.n_class*4) (300, self.n_class)  这个300只是nms后的一个过滤值,参考ProposalCreator中的参数
            roi_locs, roi_scores = self.head(features, rois)
            return roi_locs, roi_scores, rois
        # 由于batch为1所以这里直接取了第一个元素
        target_box = target_boxes[0]
        target_label = target_labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        # 为训练ROI_head 网络准备的ProposalTargetCreator
        # (128, 4)  (128, 4)     (128,)
        sample_roi, gt_head_loc, gt_head_label = self.proposal_target_creator(roi, target_box, target_label)
        # (128, self.n_class*4) (128, self.n_class)
        head_loc, head_score = self.head(features, sample_roi)

        # ------------------ 计算 RPN losses -------------------#
        # 开始计算RPN网络的定位损失
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(target_box, anchor, img_size)
        # 这里使用long类型因为下面cross_entropy方法需要
        gt_rpn_label = gt_rpn_label.long()
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        # 开始计算RPN网络的分类损失,忽略那些label为-1的
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        # ------------------计算 ROI_head losses -------------------#
        # 开始计算ROI_head网络的定位损失
        n_sample = head_loc.shape[0]
        head_loc = head_loc.reshape(n_sample, -1, 4)  # torch.Size([128, self.n_class, 4])
        # 该一步主要是获取sample_roi中每个roi所对应的修正系数loc.当然,正样本和负样本所获取的loc情况是不同的
        # 正样本:某个roi中类别概率最大的那个类别的loc;负样本:永远是第1个loc(背景类 index为0)
        gt_head_label = gt_head_label.long()
        head_loc = head_loc[torch.arange(n_sample).long().cuda(), gt_head_label]
        # 开始计算ROI_head网络的定位与分类损失
        roi_loc_loss = _fast_rcnn_loc_loss(head_loc, gt_head_loc, gt_head_label, self.roi_sigma)
        roi_cls_loss = F.cross_entropy(head_score, gt_head_label.cuda())
        losses = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        return losses

    @torch.no_grad()
    def predict(self, imgs,sizes=None):
        """
        该方法在非训练阶段的时候使用
        :param imgs: 一个batch的图片
        :param sizes: batch中每张图片的输入尺寸
        :return: 返回所有一个batch中所有图片的坐标,类,类概率值 三个值都是list型数据,里面包含的是numpy数据
        """
        boxes = list()
        labels = list()
        scores = list()
        # 因为batch_size为1所以这个循环就只循环一次
        for img, size in zip([imgs], [sizes]):
            scale = img.shape[3] / size[1]
            # (300, self.n_class*4) (300, self.n_class) (300, 4) 理论上是这样的数据,有时候可能会小于300
            roi_locs, roi_scores, roi = self(img, scale=scale)
            # chenyun版本的代码中是有对训练阶段的roi_locs进行归一化的,然后再在非训练状态下进行逆向归一化
            roi_locs = (roi_locs * self.std + self.mean)  # 减均值除以方差的逆过程

            roi_locs = roi_locs.view(-1, self.n_class, 4)  # [300, self.n_class*4] -> [300, self.n_class, 4]
            roi = roi.view(-1, 1, 4).expand_as(roi_locs)   # [300, 1, 4] -> [300, self.n_class, 4]
            # 将坐标放缩会原始尺寸 chenyun版本是将缩放这一步放到修正坐标之前,我觉得不太合理,就移到修正之后了.精度没变
            pred_boxes = loc2box_torch(roi.reshape(-1, 4),roi_locs.reshape(-1, 4)) / scale
            pred_boxes = pred_boxes  # torch.Size([5700, 4])
            pred_boxes = pred_boxes.view(-1, self.n_class, 4)   # (300*self.n_class, 4) -> (300, self.n_class, 4)
            # 限制预测框的坐标范围
            pred_boxes[:,:, 0::2].clamp_(min=0, max=size[0])
            pred_boxes[:,:, 1::2].clamp_(min=0, max=size[1])
            # 对roi_head网络预测的每类进行softmax处理
            pred_scores = F.softmax(roi_scores, dim=1)
            
            # 每张图片的预测结果(m为预测目标的个数)     # (m, 4)  (m,)  (m,) 跳过cls_id为0的pred_bbox与pred_scores,因为它是背景类
            pred_boxes, pred_label, pred_score = self._suppress(pred_boxes[:,1:,:], pred_scores[:,1:])
            boxes.append(pred_boxes)
            #   [array([[302.97562, 454.60007, 389.80545, 504.98404],
            #           [304.9767 , 550.0696 , 422.17258, 620.1692 ],
            #           [375.89203, 540.1559 , 422.39435, 684.8439 ],
            #           [293.0167, 349.53333, 360.0981, 386.8974]], dtype = float32)]
            labels.append(pred_label)
            #   [array([ 0,  0,  15, 15])]
            scores.append(pred_score)
            #   [array([0.80108094, 0.80108094, 0.80108094, 0.80108094], dtype=float32)]
        return boxes, labels, scores

    def _suppress(self, pred_boxes, pred_scores):
        """
         _suppress流程:主要是对Faster-RCNN网络最终预测的box与score进行score筛选以及nms
         1.循环所有的标注类,在循环中过滤出那些类得分在self.score_thresh之下的cls_box与cls_score。
         2.随后进行batch_nms.随后就将经过nms筛选的box,score以及新建的label分别整合到一起并返回这三个值
         :param pred_bbox: rpn网络提供的roi,经过roi_head网络提供的loc再次修正得到的 torch.Size([300, self.n_class, 4])
         :param pred_scores: roi_head网络提供各个类的置信度 torch.Size([300, self.n_class])
         :return: faster-rcnn网络预测的目标框坐标,种类,种类的置信度
         """
        cls_ids = torch.arange(self.n_class-1)[None].repeat(pred_boxes.shape[0], 1)
        # 首先过滤掉那些类得分低于self.score_thresh的
        score_keep = pred_scores > self.score_thresh
        pred_boxes = pred_boxes[score_keep].reshape(-1,4)
        pred_scores = pred_scores[score_keep].flatten()
        cls_ids = cls_ids[score_keep].flatten()
        # 这里使用batch_nms速度会快一些,A100上 35/sec -> 41/sec
        keep = batched_nms(pred_boxes, pred_scores,cls_ids, self.nms_thresh)
        box = pred_boxes[keep].cpu().numpy()
        score = pred_scores[keep].cpu().numpy()
        label = cls_ids[keep].cpu().numpy()
        return box, label, score

    def get_optimizer(self):
        # 获取梯度更新的方式,以及 放大 对网络权重中 偏置项 的学习率
        lr = cfg.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.weight_decay}]
        if cfg.use_sgd:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(params)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    def save(self, save_path=None):
        save_dict = dict()
        save_dict['model'] = self.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        save_path = 'map_%.4f.pt' % save_path

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        return self


class RegionProposalNetwork(nn.Module):
    """
    rpn网络的作用就是把由vgg特征提取网络传过来的特征进行分类和定位.
    然后利用定位卷积得出的修正系数来修正基础anchor得到初始的roi.再限制roi的坐标范围,剔除宽高小于min_size的roi.
    然后截取前12000个进行nms.最后截取前2000个roi返回
    返回的参数有:rpn预测的conf置信度,修正系数,roi,roi的batch索引,基础anchor
    """
    def __init__(self):
        super(RegionProposalNetwork, self).__init__()
        # 生成9种中心坐标为(8,8)的不同长宽不同面积的anchor坐标
        self.feat_stride = 16
        self.ratios = torch.Tensor([0.5, 1, 2])
        self.anchor_scales = torch.Tensor([8, 16, 32])
        self.anchor_base = None
        self.generate_anchor_base_torch()  # 生成9种基础anchor
        self.roi_creator = ROICreator(self)
        self.anchor_types = self.anchor_base.shape[0]
        # 初始化RPN网络的三个卷积层及其权重
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.score = nn.Conv2d(512, self.anchor_types * 2, 1, 1, 0)
        self.loc = nn.Conv2d(512, self.anchor_types * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        batch_size, channels, hh, ww = x.shape
        # 将整个输入图片内都布满anchor
        anchor = self.create_anchor_all_torch(hh, ww)
        x = F.relu(self.conv1(x))
        rpn_locs = self.loc(x)  # batch_size,36,h,w
        rpn_scores = self.score(x)  # batch_size,18,h,w
        # 对rpn网络返回的结果进行reshape
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1)
        # 这里使用softmax感觉不太合适 换用sigmoid可能会更好?
        rpn_softmax_scores = F.softmax(rpn_scores.reshape(batch_size, hh, ww, self.anchor_types, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1]  # 取第二个值为前景概率
        rpn_fg_scores = rpn_fg_scores.view(batch_size, -1)
        rpn_scores = rpn_scores.reshape(batch_size, -1, 2)
        rois = list()
        roi_indices = list()
        for i in range(batch_size):
            # roi_creator:利用rpn_loc与基础anchor得到roi,限制roi的xywh范围,
            # 按rpn_fg_scores大小截取前n个roi进行nms,截取前m个roi返回(n,m在训练与测试时不同)
            roi = self.roi_creator(rpn_locs[i].detach(),  # 这里要截断梯度
                                      rpn_fg_scores[i].detach(),
                                      anchor, img_size,scale=scale)
            # batch_index = i * np.ones((len(roi),), dtype=np.int32)  # 这里本是为了roi的bs索引准备的,但由于bs固定为1,所以省略了
            rois.append(roi)
            # roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)
        # roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

    def generate_anchor_base_torch(self):
        """
        生成基础的9种长宽、面积比的anchor坐标 坐标形式x1y1x2y2
        feat_stride: 特征提取网络下采样的倍数,这里默认是vgg16
        ratios: 三种长宽比
        anchor_scales: 和 base_size组成三种面积 (16*8)**2 (16*16)**2 (16*32)**2 意味着最大anchor在原图中的面积为512*512
        生成9种基础anchor
        """
        ratio_num = len(self.ratios)
        scale_num = len(self.anchor_scales)

        py = self.feat_stride / 2.
        px = self.feat_stride / 2.
        self.anchor_base = torch.zeros((ratio_num * scale_num, 4), dtype=torch.float32,device='cuda')
        for i in range(ratio_num):
            for j in range(scale_num):
                h = self.feat_stride * self.anchor_scales[j] * torch.sqrt(self.ratios[i])
                w = self.feat_stride * self.anchor_scales[j] * torch.sqrt(1. / self.ratios[i])
                # 每个特征点是基于box中心进行生成anchor的
                index = i * len(self.anchor_scales) + j
                self.anchor_base[index, 0] = py - h / 2.
                self.anchor_base[index, 1] = px - w / 2.
                self.anchor_base[index, 2] = py + h / 2.
                self.anchor_base[index, 3] = px + w / 2.

    def create_anchor_all_torch(self, feature_h, feature_w):
        """
        生成相对于整张图片来说的全部anchors
        :param feature_h:经过特征提取网络之后的features的高
        :param feature_w:经过特征提取网络之后的features的宽
        :return:布满整张图片的所有anchors
        """
        shift_y = torch.arange(0, feature_h * self.feat_stride, self.feat_stride, dtype=torch.float32,device='cuda')
        shift_x = torch.arange(0, feature_w * self.feat_stride, self.feat_stride, dtype=torch.float32,device='cuda')
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x,indexing='ij')
        # 这里生成的是左上角和右下角的坐标都在每个特征点左上角(需要后面拉伸开来)共(feature_h*feature_w)个anchor的坐标(yxyx形式)
        # 后面加上以特征点为中心点并且有不同面积长宽比的anchor坐标之后就成了完整的分布在fetures中的anchors
        shift = torch.stack((torch.flatten(shift_y), torch.flatten(shift_x)), 1).repeat(1, 2)
        anchor = self.anchor_base + shift[:, None, :]  # (9,4) + (1850, 1, 4) = (1850, 9, 4) np.float32
        anchor = anchor.reshape((-1, 4))  # (16650, 4)
        return anchor


class RoIHead(nn.Module):
    def __init__(self, n_class, classifier):
        super(RoIHead, self).__init__()
        self.classifier = classifier
        self.loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        # RoIHead网络的权重初始化
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        self.n_class = n_class
        self.roi = RoIPool((7, 7), 1 / 16)
        
    def forward(self, x, rois):
        """
        :param
            x           :vgg16网络提取的特征               -> torch.Size([1, 512, 37, 50]) 这里的37和50是会随输入尺寸而变化的
            rois        : RPN网络提供的roi                -> (128,4)
        return:
            roi_locs    : RoIHead网络提供的roi修正系数     -> torch.Size([128, self.n_class*4])
            roi_scores  : RoIHead网络提供的roi各类置信度   -> torch.Size([128, self.n_class])
        """
        # 这里也可同样使用官方的roi_pool来代替AdaptiveMaxPool2d,精度没什么变化但是速度变快
        rois = torch.cat((torch.zeros((rois.shape[0], 1), device='cuda'), rois), 1)
        rois = rois[:, [0, 2, 1, 4, 3]]  # ind, y x y x -> ind x y x y
        pool = self.roi(x, rois)
        pool = pool.reshape(pool.shape[0], -1)  # torch.Size([128, 25088])
        fc7 = self.classifier(pool)             # torch.Size([128, 4096])
        roi_locs = self.loc(fc7)
        roi_scores = self.score(fc7)
        return roi_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


def _smooth_l1_loss(x, t, in_weight, sigma):
    """
    这里这个sigma相当于_smooth_l1_loss在L1与L2之间切换的阈值,
    sigma为1时(默认)  smooth_l1_loss = 0.5x^2              |x| < 1
                     smooth_l1_loss = |x|-0.5             |x| >= 1
    sigma为3时,      smooth_l1_loss = 0.5x^2 * sigma^2    |x| < 1/sigma^2
                    smooth_l1_loss = |x|- 0.5/sigma^2   |x| >= 1/sigma^2
    在该份代码中计算rpn网络的损失时 sigma为3,roi网络的损失时 sigma为1
    """
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    """
    faster-rcnn中修正系数loc的损失计算
    :param pred_loc:    (h*w*9,4)    rpn网络中定位卷积提供的修正系数
    :param target_loc:  (h*w*9,4)    目标修正系数
    :param gt_label:    (h*w*9,)     默认值为-1,正样本1负样本0(共256个)的全体anchor的label值
    :param sigma:                    调整l1与l2损失函数切换的关键系数
    :return:loss                     loss计算结果
    """
    # 计算loss时只让正样本所在的权重值为1,其他默认为0.即不参与loc的loss计算
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).reshape(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss
