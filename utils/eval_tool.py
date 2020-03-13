from collections import defaultdict
import numpy as np
from utils.bbox_tools import box_iou


def eval_detection_voc(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
    # 计算precision与recall
    p, r, ap, f1, cls = calc_pr(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=iou_thresh)
    return p, r, ap, f1, cls,


def calc_pr(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
    # 将list类型的数据转换成迭代器类型的数据,注意该种类型的数据一旦经过被取出就会从迭代器中消失.是"一次性"的数据
    pred_boxes = iter(pred_boxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_boxes = iter(gt_boxes)
    gt_labels = iter(gt_labels)
    # 创建三个个默认值分别为 0 [] []的字典
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)  # 验证每类中pred_box是否为TP,是则添加一个1否则添加一个0
    # 所有测试图片中预测的情况
    for pred_box, pred_label, pred_score, gt_box, gt_label in zip(pred_boxes, pred_labels, pred_scores, gt_boxes,gt_labels):
        # 每张测试图片中预测的情况,这里只针对测试集中出现的label进行计算mAP
        for l in np.unique(gt_label):
            # 每张测试图片中l类上的预测数据
            pred_mask_l = pred_label == l
            pred_box_l = pred_box[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            score[l].extend(pred_score_l)

            # 获取在l类上的真实标注数据
            gt_box_l = gt_box[gt_label == l]
            # 代表了所有测试图片里每个类的gt_box有多少个
            n_pos[l] += gt_box_l.shape[0]
            # 如果图片中有某个类,但是在预测目标中却没有出现这个类,则跳过
            if len(pred_box_l) == 0:
                continue

            iou = box_iou(pred_box_l, gt_box_l)  # 这里的pred_box_l是经过pred_score_l从大到小排序过的
            # 获取每个pred_box与多个gt_box_l最大iou的索引
            gt_index = iou.argmax(axis=1)
            # 那些iou小于iou_thresh的索引设为-1
            gt_index[iou.max(axis=1) < iou_thresh] = -1

            # 这里代表了某个类下的所有gt_box出现状态,True代表已出现,False代表未出现.默认为False。一个gt_box只能出现一次
            selec = np.zeros(gt_box_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                # 在某一个类中,如果发现一个pred_box_l和所有gt_box_l的最大iou大于iou_thresh
                # 并且这个最大iou的gt_box_l没出现过的话则算这个pred_box_l为TP,否则记为FP
                if gt_idx >= 0 and not selec[gt_idx]:
                    match[l].append(1)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
    ap, p, r = [], [], []
    cls = list(n_pos.keys())
    cls.sort()
    for l in cls:
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        # 将预测的所有图片中某类下面的score从大到小排序,计算map第一步
        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)
        # 同上如果有标注目标,但是没有预测目标这会导致tp和fp都为np.array([])进一步导致prec[l]和rec[l]均为np.array([])
        if tp.size == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            recall_curve = tp / n_pos[l]
            r.append(recall_curve[-1])
            precision_curve = tp / (fp + tp + 1e-8)
            p.append(precision_curve[-1])
            ap.append(calc_ap(precision_curve, recall_curve))
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, ap, f1, cls


def calc_ap(prec, rec, interpolated_ap=False):
    if interpolated_ap:
        # Voc 2010年以前使用的计算mAP的方法:11-point interpolated average precision(11点插值面积法)
        # 该种计算mAP的方法是把整个PR曲线分为11个部分,然后计算均值
        ap = 0
        for t in np.arange(0., 1.1, 0.1):
            # 这个if可以处理prec[l]和rec[l]均为np.array([])的情况
            if np.sum(rec >= t) == 0:
                p = 0
            # 到这一步,prec[l])和rec[l]就必不可能是np.array([])或者None的情况了,就可以正常的计算precision了
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11
    else:
        # Voc 2010年及以后使用的计算mAP的方法:积分面积法
        # 为了方便计算AP的值,我们需要现在每类的PR值中加入两个临时值,(这些值不影响后面AP的计算)
        mpre = np.concatenate(([0], prec, [0]))
        mrec = np.concatenate(([0], rec, [1]))
        # 这一步的操作先将mpre从大到小转为从小到大,然后从小到大取自己和左边的值之间的最大值(结合np.maximum.accumulate体会)
        # 其实这一步主要是将PR曲线由折线型抹平成阶梯型来方便计算AP,最后再转回从大到小排序
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]

        # 开始计算AP的面积,寻找recall值变动的位置.多个低*高 加起来就是AP的值
        i = np.where(mrec[:-1] != mrec[1:])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

