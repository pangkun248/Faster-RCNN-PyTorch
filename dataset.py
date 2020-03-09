from torch.utils.data import Dataset
from torchvision import transforms as tvtsf
import numpy as np
from PIL import Image
import torch.nn.functional as F
import random
import glob

class ListDataset(Dataset):
    def __init__(self,path,is_train=True):
        self.is_train = is_train
        with open(path, 'r') as file:
            self.img_paths = file.readlines()
        # 根据图片的路径得到 label 的路径, label 的存储格式为一个图片对应一个.txt文件
        # 文件的每一行代表了该图片的 box 信息, 其内容为: class_id, x, y, w, h (xywh都是用小数形式存储的,相对坐标)
        self.label_paths = [path.replace('JPGImages', 'labels').replace('.jpg', '.txt') for path in self.img_paths]

    def __getitem__(self, index):
        # 这里必须要加一个rstrip()去除txt文件每行末尾的\n换行,不然文件名会出错导致系统找不到路径,同理图片路径也是一样
        label_path = self.label_paths[index].strip()
        # label_data -> cls_id, ymin, xmin, ymax, xmax 绝对坐标
        label_data = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
        label = label_data[:, 0].astype(int)
        box = label_data[:, 1:]

        # 上面是加载标签数据,下面加载图片数据以及相应的处理

        img_path = self.img_paths[index].strip()
        # 1.Image.open -> ndarray 注:PIL打开图片的方式默认为RGB格式,不需要再额外转换
        # 2.ToTensor -> transpose(2, 0, 1)
        #            -> torch.from_numpy
        #            -> .div(255)
        img = tvtsf.ToTensor()(Image.open(img_path))    # torch.float32
        in_c, in_h, in_w = img.shape
        # img = preprocess(img)
        # 缩放到最小比例,这样最终长和宽都能放缩到规定的尺寸
        scale1 = 600 / min(in_h, in_w)
        scale2 = 1000 / max(in_h, in_w)
        scale = min(scale1, scale2)
        # resize到最小比例,anti_aliasing为是否采用高斯滤波 使用sk-learn的方式来resize
        # img = sktsf.resize(img, (in_c, in_h * scale, in_w * scale), mode='reflect', anti_aliasing=False)  # np.float64

        img = F.interpolate(img.unsqueeze(0), size=(round(in_h * scale), round(in_w * scale)), mode="nearest").squeeze(0)
        # transforms.Normalize使用如下公式进行归一化 value=(value-mean)/std,转换为[-1,1],caffe只有减去均值
        img = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)  # torch.float32
        out_c, out_h, out_w = img.shape
        if self.is_train:
            box *= scale
            # 水平翻转 目前的任务场景中不太需要此功能
            # 但是如果开启此功能,需要将后续返回的img替换为img.copy()
            # 因为给定numpy数组的某些步幅为负(img[:, ::-1, :]和img[:, :, ::-1])。numpy官方当前不支持此功能,但将来的版本中将添加此功能。
            # img, params = util.random_flip(img, x_random=True, return_param=True)
            # bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
            return img, box, label, scale
        else:
            return img, (out_h, out_w), box, label

    def __len__(self):
        return len(self.img_paths)


# 为测试图片准备
class ImageFolder(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob("%s/*.*" % folder_path)

    def __getitem__(self, index):
        img_path = self.files[index]
        # Extract image as PyTorch tensor
        img = tvtsf.ToTensor()(Image.open(img_path))
        in_c, in_h, in_w = img.shape
        # img = preprocess(img)
        # 缩放到最小比例,这样最终长和宽都能放缩到规定的尺寸
        scale1 = 600 / min(in_h, in_w)
        scale2 = 1000 / max(in_h, in_w)
        scale = min(scale1, scale2)
        img = F.interpolate(img.unsqueeze(0), size=(round(in_h * scale), round(in_w * scale)), mode="nearest").squeeze(0)
        img = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img_path, img, img.shape[1:]

    def __len__(self):
        return len(self.files)


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """
    这里需要说明一下,box是如何进行水平或竖直翻转的
    拿x方向水平翻转举例:根据对称性,翻转后box右边框的x坐标等于图像整体宽度减去翻转前box左边框的x坐标.
                 同理:翻转后box左边框的x坐标等于图像整体宽度减去翻转前box右边框的x坐标.
    y方向竖直翻转同理.
    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def crop_bbox(bbox, y_slice=None, x_slice=None, allow_outside_center=True, return_param=False):

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
            .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def random_flip(img, y_random=False, x_random=False, return_param=False, copy=False):
    """
    :param img: numpy型的数组 CHW格式
    :param y_random: 是否进行竖直方向翻转
    :param x_random: 是否进行水平方向翻转
    :param return_param: 是否返回水平竖直方向翻转信息
    :param copy: 是否创建一个新的img进行翻转
    :return:取决于return_param,如果为false,则只返回翻转后的img,否则返回翻转后的img和翻转信息
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img
