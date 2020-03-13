from tqdm import tqdm
from config import opt
from dataset import ListDataset
from model import FasterRCNN
from torch.utils.data import DataLoader
from utils import array_tool as at
from utils.eval_tool import eval_detection_voc
from terminaltables import AsciiTable
import visdom
import numpy as np
import torch


def eval(dataloader, model):
    pred_boxes, pred_labels, pred_scores = list(), list(), list()
    gt_boxes, gt_labels = list(), list()
    with torch.no_grad():
        for imgs, sizes, gt_boxes_, gt_labels_ in tqdm(dataloader):
            sizes = [sizes[0].item(), sizes[1].item()]
            pred_boxes_, pred_labels_, pred_scores_ = model.predict(imgs, [sizes])
            gt_boxes += list(gt_boxes_.numpy())
            gt_labels += list(gt_labels_.numpy())
            pred_boxes += pred_boxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_
    result = eval_detection_voc(pred_boxes, pred_labels, pred_scores,gt_boxes, gt_labels)
    return result


if __name__ == '__main__':
    # 准备训练与验证数据
    dataset = ListDataset(opt.train_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,num_workers=opt.num_workers)
    testset = ListDataset(opt.train_dir, is_train=False)
    test_dataloader = DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=False,pin_memory=True)
    # 加载模型与权重
    model = FasterRCNN().cuda()
    if opt.load_path:
        model.load(opt.load_path)
    # 创建visdom可视化端口
    vis = visdom.Visdom(env='Faster R-CNN')
    best_map = 0
    for epoch in range(1,opt.epoch):
        for img, target_box, target_label, scale in tqdm(dataloader):
            scale = at.scalar(scale)
            img, target_box, target_label = img.cuda().float(), target_box.cuda(), target_label.cuda()
            model.optimizer.zero_grad()
            loss = model(img, target_box, target_label, scale)
            loss['total_loss'].backward()
            model.optimizer.step()
        # 每个Epoch计算一次mAP
        ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        eval_result = eval(test_dataloader, model)
        for p, r, ap, f1, cls_id in zip(*eval_result):
            ap_table += [[cls_id, opt.class_name[cls_id], "%.3f" % p, "%.3f" % r, "%.3f" % ap, "%.3f" % f1]]
        print('\n' + AsciiTable(ap_table).table)
        eval_map = round(eval_result[2].mean(),4)
        print("Epoch %d/%d ---- mAP:%.4f Loss:%.4f" % (epoch, opt.epoch, eval_map, loss['total_loss']))
        # 绘制mAP和Loss曲线
        vis.line(X=np.array([epoch]), Y=np.array([eval_map]), win='mAP',
                 update=None if epoch == 1 else 'append', opts=dict(title='mAP'))
        vis.line(X=np.array([epoch]), Y=torch.tensor([loss['total_loss']]), win='Loss',
                 update=None if epoch == 1 else 'append', opts=dict(title='Loss'))
        # 保存目前最佳模型,每隔指定Epoch加载最佳模型,同时下调学习率
        if eval_map > best_map:
            best_map = eval_map
            best_path = model.save(save_path=str(best_map))
        if epoch == 9:
            model.load(best_path)
            model.scale_lr(opt.lr_decay)
