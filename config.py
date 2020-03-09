class Config:
    # 训练图片路径,验证图片路径(mAP相关),最终测试图片路径
    train_dir = r'E:\Faster RCNN-PyTorch\data\wenyi\train.txt'
    val_dir = r'E:\Faster RCNN-PyTorch\data\wenyi\val.txt'
    test_dir = r'E:\Faster RCNN-PyTorch\data\wenyi\test'
    # 图片最大与最小输入长宽尺寸
    max_size = 1000
    min_size = 600
    num_workers = 6
    test_num_workers = 6

    # 计算loss时rpn与roi更新的幅度
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    env = 'faster-rcnn'  # visdom env
    epoch = 14

    use_adam = False # Use Adam optimizer
    load_path = None

    class_name = ("WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster", "MiniZalu",
    "Dopelliwin","ShieldAxe", "SkeletonKnight","Zalu","Cyclone","SlurryBeggar","Gerozaru","Catalog",
    "InfectedMonst","Gold","StormRider","Close","Door",)


opt = Config()
