from utils.box_tools import loc2box, box2loc, box_iou, _get_inside_index, _unmap, NMS
import numpy as np
from config import cfg

class ProposalCreator:
    """
    ProposalCreator的主要功能如下:
    1.由rpn得出的修正系数来修正基础anchor来得到roi
    2.限制roi的坐标范围
    3.剔除那些宽高小于min_size的roi
    4.根据rpn分类卷积得出的conf来从大到小进行排序,并截取前 12000个roi(如果少于12000,如8000.那么就截取前8000.下面同理)
    5.进行nms,然后截取前 2000个roi.并最终返回这些roi(训练阶段->非训练阶段,nms前12000->6000,nms后2000->300)
    """
    def __init__(self,parent_model,):
        self.parent_model = parent_model
        # 下面四个数字可以缩小10倍,而对精度影响不是那么大.但是训练和测试速度却可以翻倍
        self.n_train_pre_nms = 1200
        self.n_train_post_nms = 200
        self.n_test_pre_nms = 600
        self.n_test_post_nms = 30
        self.min_size = 16

    def __call__(self, loc, score,anchor, img_size, scale=1.,is_train=True):
        """
        :param loc: rpn网络定位卷积得出的修正参数    (16650, 4)
        :param score: rpn网络分类卷积得出的置信度    (16650,)
        :param anchor: 基础生成的anchor            (16650, 4)
        :param img_size: 网络输入的尺寸 (h,w)
        :param scale: 原始图片resize到网络输入尺寸的倍数
        :return: 经过筛选的roi 当然也可以多返回一个pred_box为是否含有物体的目标置信度
        """
        if is_train:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        roi = loc2box(anchor, loc)  # (16650, 4)
        # 限制roi的坐标范围
        roi[:, 0:4:2] = np.clip(roi[:, 0:4:2], 0, img_size[0])
        roi[:, 1:4:2] = np.clip(roi[:, 1:4:2], 0, img_size[1])
        # 剔除那些宽高小于min_size的roi
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep]
        score = score[keep]
        # 重新根据分类置信度从大到小进行排序
        order = score.argsort()[::-1]
        order = order[:n_pre_nms]
        roi = roi[order]
        score = score[order]
        roi,score = NMS(roi, score, cfg.nms_rpn)
        roi = roi[:n_post_nms]
        return roi


class AnchorTargetCreator(object):
    """
    就像类名一样 生成Anchor对应的Target Anchor是本来就存在的
    在训练RPN网络时,需要准备一些target_loc和target_label来和rpn网络生成rpn_loc和rpn_label来计算rpn网络的损失.
    AnchorTargetCreator 就是为此而准备的
    先从基础anchor中提取出内部框的索引 inside_index,并修改基础anchor为内部anchor.
    计算anchor与target_boxes的iou值,返回每个anchor与target_boxes的最大iou索引argmax_ious以及满足iou条件的不超过256个的正负样本索引
    创建一个默认值为 -1的长度为len(anchor)的基础label,并根据前面得到的正负样本的索引,分别给label中正样本赋1负样本赋0
    根据target_box与前一步得到的argmax_ious求出与anchor最匹配的target_box,然后求出真实修正系数
    最后将在内部anchor上求出的内部loc与内部label映射回基础loc(除内部loc外默认为0)与基础label(除内部label外默认为-1)上,
    最终返回基础loc与基础label
    """
    def __init__(self):
        self.n_sample = 256
        self.pos_iou_thresh = 0.7
        self.neg_iou_thresh = 0.3
        self.pos_ratio = 0.5

    def __call__(self, target_box, anchor, img_size):
        """
        :param target_box:真实标注的框
        :param anchor: 初步生成的基础框
        :param img_size:图片输入尺寸
        :return:rpn_box到target_box的真实矫正系数(针对所有anchor的),和是否含有目标的label(1=有, 0=无, -1=忽略)
        """
        img_h, img_w = img_size
        n_anchor = len(anchor)
        # 获取不超出图片边界的anchors索引
        inside_index = _get_inside_index(anchor, img_h, img_w)
        anchor_inside = anchor[inside_index]
        argmax_ious, label = self._create_label(anchor_inside, target_box)
        # 计算所有内部框到每个框最匹配(iou)的target_box的修正系数.
        loc = box2loc(anchor_inside, target_box[argmax_ious])
        # 将内部的label和loc 映射到完整的原始完整的label和loc中去
        label = _unmap(label, n_anchor, inside_index)
        loc = _unmap(loc, n_anchor, inside_index)

        return loc, label

    def _create_label(self, anchor, target_box):
        """
        :param anchor:内部anchors
        :param target_box:真实标注了的box
        :return: argmax_ious 每个 anchor与所有target_box 的最大iou索引;
                 label 所有anchor对应的标签 1 是正样本, 0 是负样本, -1 表示忽略
        """
        label = -np.ones((len(anchor),), dtype=np.int32)

        # 计算内框和标注框的iou
        ious = box_iou(anchor, target_box)
        argmax_ious = ious.argmax(axis=1)   # 每个 anchor与target_boxes的最大iou的索引
        max_ious = ious.max(axis=1)         # 每个 anchor与target_boxes的最大iou的值
        # 每个 target_box与所有anchors的最大iou
        gt_max_ious = ious.max(axis=0)
        # 这里gt_max_ious的最大值可能不是唯一的,所以需要把全部的最大iou都找出来作为 "target_anchor"的索引
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        # 首先分配负样本,以便正样本可以覆盖它们(某些情况下最大IOU可能小于neg_iou_thresh)
        # 负样本: iou小于 neg_iou_thresh的anchor
        label[max_ious < self.neg_iou_thresh] = 0

        # 正样本:对于每个target_box有最大IOU的某个或多个anchor
        label[gt_argmax_ious] = 1
        # 正样本: iou大于 pos_iou_thresh的anchor
        label[max_ious >= self.pos_iou_thresh] = 1
        # 如果正样本超过理论值则随机丢弃多余的正样本
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 如果负样本超过理论值则随机丢弃多余的负样本
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    """
   就像类名一样 生成Proposal对应的Target Proposal是由rpn提供的
   先把target_box并入到roi中去,计算roi和target_box的iou,并获取每个roi和target_box的最大iou索引roi_argmaxiou 以及最大iou值roi_maxiou
   将 target_box的label通过roi_argmaxiou赋值给roi,并+1(因为0为背景类)
   根据正负样本的iou阈值得出iou中的正负样本索引(共128个)pos_index,neg_index(组成keep_index).并随机舍弃多余的样本
   从众多roi_label中根据keep_index挑选出正负样本的label,并令负样本的label为0
   并根据keep_index从roi中挑选出正负样本的roi,然后由target_box根据roi_argmaxiou和keep_index得出len(keep_index)个
   与roi正负样本对应的target_box 即target_box[roi_argmax_targets[keep_index]]
   最后根据正负样本的roi与其对应的target_box计算修正系数,然后减均值除以方差
   参数:
       n_sample (int): 每张图片理论上采集的样本数.
       pos_ratio (float): 正样本比例
       pos_iou_thresh (float): 达到正样本的IOU阈值
       neg_iou_thresh_hi (float): IOU在此区间内的属于负样本 [neg_iou_thresh_lo, neg_iou_thresh_hi).
       neg_iou_thresh_lo (float): 同上.
    """
    def __init__(self,n_sample=128,pos_ratio=0.25, pos_iou_thresh=0.5,neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # 注意:在py-faster-rcnn中 该值默认为0.1

    def __call__(self, roi, target_box, label,loc_normalize_mean=(0., 0., 0., 0.),loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """
        :param roi: rpn网络提供的roi,理论上训练阶段为(2000,4) 测试阶段为(300,4)
        :param target_box: 真实标注物体的坐标 (n,4) n为一张图片中真实标注物体的个数
        :param target_label: 真实标注物体的种类索引 (n,)
        :param loc_normalize_mean: 标准化坐标的平均值
        :param loc_normalize_std: 坐标的标准方差
        :return: 128(理论)个正负样本roi的坐标, roi与target_box修正系数, 128(理论)个正负样本roi的label
        """
        # n_bbox, _ = target_box.shape
        # 这里将target_box也并入到roi中去
        roi = np.concatenate((roi, target_box), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = box_iou(roi, target_box)
        # 每个roi和tartget_boxes的最大iou索引
        gt_assignment = iou.argmax(axis=1)
        # 每个roi和n个tartget_box的最大iou
        max_iou = iou.max(axis=1)
        # 将所有种类索引+1(所有label>=1,0为下面的负样本所准备的),并且此时为所有roi赋予label.值为与其iou最大的target_box的label值
        gt_roi_label = label[gt_assignment] + 1  # (roi.shape[0],)

        # 获取那些IOU大于pos_iou_thresh的roi索引
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # 获取那些IOU在[neg_iou_thresh_lo, neg_iou_thresh_hi)区间的roi索引
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        # 计算每张图片中理论上的负样本个数
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # 将正负样本的roi索引合并到一起
        keep_index = np.append(pos_index, neg_index)
        # 从所有roi中挑选出正负样本的label
        gt_roi_label = gt_roi_label[keep_index]
        # 将负样本的label置为0
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        # 计算修正系数    roi和其最大iou的target_box的loc
        gt_roi_loc = box2loc(sample_roi, target_box[gt_assignment[keep_index]])
        # 这里的减均值除以方差以及非训练阶段roi网络最后出来的roi_loc还要乘方差加均值,看不懂为什么要这样做
        # gt_roi_loc=((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label







