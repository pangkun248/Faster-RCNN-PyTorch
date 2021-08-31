class Config:
    # 训练集路径,验证集路径(mAP相关)
    train_dir = 'VOC2007文件夹路径'
    val_dir = 'VOC2007文件夹路径'
    # 图片最大与最小输入长宽尺寸
    max_size = 1000
    min_size = 600
    num_workers = 8         # 取决于你的cpu核数
    test_num_workers = 8    # 同上

    # 计算loss时rpn与roi所占的比重
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005   # 权重衰减系数 原始论文中为0.0005 但是tf-faster-rcnn中为0.0001
    lr_decay = 0.1  # 每隔指定epoch学习率下降的倍数
    lr = 1e-3       # 初始学习率
    epoch = 14
    nms_rpn = 0.7   # RPN阶段 ROICreator 中的nms阈值  不区分是否是training
    nms_roi = 0.3   # ROI阶段的nms阈值 仅存在于非训练阶段,即训练阶段整个网络只有一个rpn阶段的nms,而非训练阶段则有两个
    use_sgd = True  # 是否使用SGD优化方式
    load_path = r''  # 基于此模型权重训练
    # 注意Faster-RCNN中是由背景这一类的,但是这里及xml2txt都没有 '__background__'这一类,是因为Faster-RCNN在内部临时把所有的
    # target_label都+1,计算ap以及最后nms的时候又跳过label等于0的情况. 详情参见 ProposalTargetCreator 类 以及_suppress方法
    class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


cfg = Config()
