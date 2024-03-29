from tqdm import tqdm
from config import cfg
from dataset import ListDataset
from model import FasterRCNN
from torch.utils.data import DataLoader
from utils.eval_tool import Eval
from utils.chen_map import eval
from terminaltables import AsciiTable
# import visdom
import numpy as np
import torch

if __name__ == '__main__':
    # 准备训练与验证数据
    trainset = ListDataset(cfg, is_train=True)
    dataloader = DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True)  # 由于bs为1所以num_work不再设置
    testset = ListDataset(cfg, split='test', is_train=False)
    test_dataloader = DataLoader(testset, batch_size=1, pin_memory=True)
    # 加载模型与权重
    model = FasterRCNN().cuda()
    if cfg.load_path:
        model.load(cfg.load_path)
        print('已加载训练模型')
    # 创建visdom可视化端口
    # vis = visdom.Visdom(env='Faster R-CNN')
    best_map = 0
    for epoch in range(cfg.epoch):
        for img, target_box, target_label, scale in tqdm(dataloader):
            scale = scale.cuda()
            img, target_box, target_label = img.cuda().float(), target_box.cuda(), target_label.cuda()
            loss = model(img, target_box, target_label, scale)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

        model.eval()
        chen_result = eval(test_dataloader, model)
        # 每个Epoch计算一次mAP
        # ap_table = [["Index", "Class name", "Precision", "Recall", "AP:0.5", "F1-score"]]
        # eval_result = Eval(test_dataloader, model)  # 在计算mAP时不会忽略difficult值为1的box
        # for p, r, ap, f1, cls_id in zip(*eval_result):
        #     ap_table += [[cls_id + 1, cfg.class_name[cls_id], "%.3f" % p, "%.3f" % r, "%.3f" % ap, "%.3f" % f1]]
        # print('\n' + AsciiTable(ap_table).table)
        # eval_map = eval_result[2].mean()
        print("Epoch %d/%d ---- chen-mAP:%.4f" % (epoch, cfg.epoch, chen_result['map']))
        # 绘制mAP和Loss曲线
        # vis.line(X=np.array([epoch]), Y=np.array([eval_map]), win='mAP',
        #        update=None if epoch == 1 else 'append', opts=dict(title='mAP'))
        # vis.line(X=np.array([epoch]), Y=torch.tensor([loss['total_loss']]), win='Loss',
        #         update=None if epoch == 1 else 'append', opts=dict(title='Loss'))
        model.train()
        # 保存目前最佳模型,每隔指定Epoch加载最佳模型,同时下调学习率
        # if chen_result['map'] > best_map:
        #    best_map = chen_result['map']
        #    best_path = model.save(save_path=best_map)
        if epoch == 9:
            # model.load(best_path)
            model.scale_lr(cfg.lr_decay)
