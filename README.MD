此代码源自https://github.com/chenyuntc/simple-faster-rcnn-pytorch

并在此基础上进行修改与注释.以下为基于VOC2007数据集的指标,

|ignore_difficult |chen_mAP|mine|
|---          |--- |---      |
|simple(default)      |0.6913|0.7467    |
|True      |0.6878| 0.7472     |
|False      |0.6876|0.6633      |
|is_train      |0.6966|0.6499        |
|not is_train      |0.6765|0.7370     |

ignore_difficult 是否忽略该属性为1的box,simple是训练忽略测不忽略,但是在计算mAP时会把difficult的list传给计计算mAP的函数,应该也会做类似忽略的处理

chen_mAP:使用作者计算mAP的方法

mine:使用我自己重写的方法计算mAP

训练集： trainval.txt 

测试集： test.txt

训练:config文件修数据集路径之后直接运行train.py即可训练
