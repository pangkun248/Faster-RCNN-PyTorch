import numpy as np


def loc2box(src_box, loc):
    """
    已知预测框和修正参数,求目标框
    利用平移和尺度放缩修正P_box以得到 ^G_box 然后将^G_box与G_box进行计算损失
    参考 https://blog.csdn.net/zijin0802034/article/details/77685438/
    ^G_box计算方式
    ^G_box_y = p_h*t_y + p_y`
    ^G_box_x = p_w*t_x + p_x`
    ^G_box_h = p_h*exp(t_h)`
    ^G_box_w = p_w*exp(t_w)`
    参数:
        src_bbox (array): `p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`. (R,4)
        loc (array): `t_y, t_x, t_h, t_w`.  (R,4)

    返回:
        修正后的^G_box->shape为(R, 4),R是预测的框数量
        第二维度的数据形式与src_bbox相同
    """
    if src_box.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_box = src_box.astype(src_box.dtype, copy=False)
    # 转换坐标格式 x1y1x2y2 -> xywh
    src_h = src_box[:, 2] - src_box[:, 0]
    src_w = src_box[:, 3] - src_box[:, 1]
    src_y = src_box[:, 0] + 0.5 * src_h
    src_x = src_box[:, 1] + 0.5 * src_w

    # 各个修正系数
    dy = loc[:, 0]
    dx = loc[:, 1]
    dh = loc[:, 2]
    dw = loc[:, 3]

    dst_y = dy * src_h + src_y
    dst_x = dx * src_w + src_x
    dst_h = np.exp(dh) * src_h
    dst_w = np.exp(dw) * src_w

    # 转换坐标格式 xywh -> x1y1x2y2
    dst_box = np.zeros(loc.shape, dtype=loc.dtype)
    dst_box[:, 0] = dst_y - 0.5 * dst_h
    dst_box[:, 1] = dst_x - 0.5 * dst_w
    dst_box[:, 2] = dst_y + 0.5 * dst_h
    dst_box[:, 3] = dst_x + 0.5 * dst_w

    return dst_box


def box2loc(src_box, dst_box):
    """
    已知真实框和预测框求出其修正参数
    :param src_box: shape -> (R, 4) x1y1x2y2.
    :param dst_box: 同上
    :return: 修正系数 shape -> (R, 4)
    """
    src_h = src_box[:, 2] - src_box[:, 0]
    src_w = src_box[:, 3] - src_box[:, 1]
    src_y = src_box[:, 0] + 0.5 * src_h
    src_x = src_box[:, 1] + 0.5 * src_w

    dst_h = dst_box[:, 2] - dst_box[:, 0]
    dst_w = dst_box[:, 3] - dst_box[:, 1]
    dst_y = dst_box[:, 0] + 0.5 * dst_h
    dst_x = dst_box[:, 1] + 0.5 * dst_w

    dy = (dst_y - src_y) / (src_h + 1e-8)
    dx = (dst_x - src_x) / (src_w + 1e-8)
    dh = np.log(dst_h / (src_h + 1e-8))
    dw = np.log(dst_w / (src_w + 1e-8))

    loc = np.stack((dy, dx, dh, dw), axis=1)
    return loc


def box_iou(box_a, box_b):
    # 计算 N个box与M个box的iou需要使用到numpy的广播特性
    # tl为交叉部分左上角坐标最大值, tl.shape -> (N,M,2)
    tl = np.maximum(box_a[:, np.newaxis, :2], box_b[:, :2])
    # br为交叉部分右下角坐标最小值
    br = np.minimum(box_a[:, np.newaxis, 2:], box_b[:, 2:])
    # 第一个axis是指定某一个box内宽高进行相乘,第二个axis是筛除那些没有交叉部分的box
    # 这个 < 和 all(axis=2) 是为了保证右下角的xy坐标必须大于左上角的xy坐标,否则最终没有重合部分的box公共面积为0
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # 分别计算bbox_a,bbox_b的面积,以及最后的iou
    area_a = np.prod(box_a[:, 2:] - box_a[:, :2], axis=1)
    area_b = np.prod(box_b[:, 2:] - box_b[:, :2], axis=1)
    iou = area_i / (area_a[:, np.newaxis] + area_b - area_i)
    return iou


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],anchor_scales=[8, 16, 32]):
    """
    生成基础的9种长宽、面积比的anchor坐标 坐标形式x1y1x2y2
    :param base_size: 特征提取网络下采样的倍数
    :param ratios: 三种长宽比
    :param anchor_scales: 和 base_size组成三种面积 (16*8)**2 (16*16)**2 (16*32)**2 意味着最大anchor在原图中的面积为512*512
    :return:生成好的9种基础anchor
    """
    py = base_size / 2.
    px = base_size / 2.
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            # 每个特征点是基于box中心进行生成anchor的
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def create_anchor_all(anchor_base, feat_stride, height, width):
    """
    生成相对于整张图片来说的全部anchors
    :param anchor_base: 9种基础的anchor坐标
    :param feat_stride:下采样倍数
    :param height:图片输入高度
    :param width:图片输入高度
    :return:布满整张图片的所有anchors
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), axis=1)
    # A代表了基础anchor的个数 9
    A = anchor_base.shape[0]
    # K代表了一张图片中有多少个anchor中心 每个anchor中心可以有9中anchor
    K = shift.shape[0]
    # 这里用了numpy的广播机制
    anchor = anchor_base + shift[:, np.newaxis, :]  # (1850, 9, 4)  np.float64
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  # (16650, 4)  np.float32
    return anchor


def _get_inside_index(anchor, h, w):
    # 返回那些在指定大小的图片内部的anchor索引
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= h) &
        (anchor[:, 3] <= w)
    )[0]
    return index_inside


def _unmap(data, n_anchor, inside_index):
    if len(data.shape) == 1:
        # 如果是对label进行映射,则默认值为-1(忽略样本)
        ret = -np.ones((n_anchor,), dtype=np.int32)
        ret[inside_index] = data
    else:
        # 如果是对loc进行映射,则默认值为0(忽略样本)
        ret = np.zeros((n_anchor, 4), dtype=np.float32)
        ret[inside_index] = data
    return ret


def NMS(box,score, nms_thres):
    keep_boxes = []
    keep_scores = []
    while box.shape[0]:
        yy1 = np.maximum(box[0, 0], box[:, 0])
        xx1 = np.maximum(box[0, 1], box[:, 1])
        yy2 = np.minimum(box[0, 2], box[:, 2])
        xx2 = np.minimum(box[0, 3], box[:, 3])

        # 计算重叠区域
        inter_area = np.maximum(yy2 - yy1 + 1,0) * np.maximum(xx2 - xx1 + 1,0)
        # 计算每个box的面积
        box_area = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
        iou = inter_area / (box_area[0] + box_area - inter_area)
        cut_index = iou > nms_thres
        keep_boxes += [box[0]]
        keep_scores += [score[0]]
        box = box[~cut_index]
        score = score[~cut_index]
    keep_boxes = np.stack(keep_boxes)
    keep_scores = np.stack(keep_scores)
    return keep_boxes,keep_scores