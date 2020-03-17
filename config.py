class Config:
    # 训练集路径,验证集路径(mAP相关)
    train_dir = r'D:\py_pro\Faster-RCNN-PyTorch\data\wenyi\train.txt'
    val_dir = r'D:\py_pro\Faster-RCNN-PyTorch\data\wenyi\val.txt'
    # 图片最大与最小输入长宽尺寸
    max_size = 1000
    min_size = 600
    num_workers = 6         # 取决于你的cpu核数
    test_num_workers = 6    # 同上

    # 计算loss时rpn与roi所占的比重
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005   # 权重衰减系数 原始论文中为0.0005 但是tf-faster-rcnn中为0.0001
    lr_decay = 0.1  # 每隔指定epoch学习率下降的倍数
    lr = 1e-3       # 初始学习率
    epoch = 14

    use_adam = False # 是否使用Adam优化方式
    load_path = r'D:\py_pro\Faster-RCNN-PyTorch\weights\map_0.9208.pt'  # 基于此模型权重训练
    class_name = ("WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster", "MiniZalu",
    "Dopelliwin","ShieldAxe", "SkeletonKnight","Zalu","Cyclone","SlurryBeggar","Gerozaru","Catalog",
    "InfectedMonst","Gold","StormRider","Close","Door",)


opt = Config()
