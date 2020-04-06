class Config:
    # 训练集路径,验证集路径(mAP相关)
    train_dir = r'D:\py_pro\Faster-RCNN-PyTorch\data\wenyi\train.txt'
    val_dir = r'D:\py_pro\Faster-RCNN-PyTorch\data\wenyi\val.txt'
    # 图片最大与最小输入长宽尺寸
    max_size = 1000
    min_size = 600
    num_workers = 2         # 取决于你的cpu核数
    test_num_workers = 2    # 同上

    # 计算loss时rpn与roi所占的比重
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005   # 权重衰减系数 原始论文中为0.0005 但是tf-faster-rcnn中为0.0001
    lr_decay = 0.1  # 每隔指定epoch学习率下降的倍数
    lr = 1e-3       # 初始学习率
    epoch = 14
    nms_rpn = 0.7   # rpn阶段 ProposalCreator 中的nms阈值
    nms_test = 0.3  # 非训练阶段的nms阈值 用于筛选Faster-RCNN给出的pred_boxes
    use_adam = False # 是否使用Adam优化方式
    load_path = r''  # 基于此模型权重训练
    # 注意Faster-RCNN中是由背景这一类的,但是这里及xml2txt都没有 '__background__'这一类,是因为Faster-RCNN在内部临时把所有的
    # target_label都+1,计算ap以及最后nms的时候又跳过label等于0的情况. 详情参见 ProposalTargetCreator 类 以及_suppress方法
    class_name = ("WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster", "MiniZalu",
    "Dopelliwin","ShieldAxe", "SkeletonKnight","Zalu","Cyclone","SlurryBeggar","Gerozaru","Catalog",
    "InfectedMonst","Gold","StormRider","Close","Door",)


cfg = Config()
