import torch
from model import FasterRCNN
from config import opt
import colorsys
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
from dataset import ImageFolder


dection_imgs = ImageFolder(r'D:\py_pro\Faster-RCNN-PyTorch\data\wenyi\test')
model = FasterRCNN().cuda()
model.load_state_dict(torch.load(r'D:\py_pro\Faster-RCNN-PyTorch\weights\map_0.9208.pt')['model'])
model.eval()

# 为每个类名配置不同的颜色
cls_name = opt.class_name
hsv_tuples = [(x / len(cls_name), 1., 1.)for x in range(len(cls_name))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

imgs_path = []  # 测试图片保存路径
img_detections = []  # 每张图片的检测结果
imgs_size = []

# 先保存检测所得结果
for path, img,size in dection_imgs:
    img = img.unsqueeze(0)
    imgs_path.append(path)
    with torch.no_grad():
        pred_boxes_, pred_labels_, pred_scores_ = model.predict(img, [size])
        imgs_size.append(size)
        img_detections.append([pred_boxes_, pred_labels_, pred_scores_])

# 然后对所有图片进行处理
for path,img_detection,size in zip(imgs_path,img_detections,imgs_size):
    PIL_img = Image.open(path)
    w,h = PIL_img.size
    content_font = ImageFont.truetype(font='FiraMono-Medium.otf', size=16)
    thickness = (w + h) // 600
    draw = ImageDraw.Draw(PIL_img)
    # 对单张图片进行处理
    for box,label,score in zip(*img_detection):
        box, label, score = box.tolist(), label.tolist(), score.tolist()
        # 对一张图片中预测的每个box进行处理
        for (y1,x1,y2,x2),l,s in zip(box,label,score):
            # 对预测出的坐标进行缩放
            y1, x1, y2, x2 = y1*(w/size[1]), x1*(w/size[1]), y2*(w/size[1]), x2*(w/size[1])
            content = '{} {:.2f}'.format(opt.class_name[l], s)
            label_w, label_h = draw.textsize(content,content_font)
            for i in range(thickness):
                draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=colors[l])
                draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=colors[l])
                draw.text((x1, y1 - label_h), content, fill=(0, 0, 0),font=content_font)
        PIL_img = np.array(PIL_img)[...,::-1]
        cv2.imshow('result',PIL_img)
        cv2.waitKey(0)
