此代码源自https://github.com/chenyuntc/simple-faster-rcnn-pytorch

并在此基础上进行修改与注释.以下为基于VOC2007数据集的指标,(主干网络为vgg16)

|ignore_difficult |mAP|
|---          |--- |
|True      |0.6966|
|False      |0.6876|

ignore_difficult: 是否忽略difficult值为1的box

True:训练测试都忽略,False:都不忽略


训练集： 2007/trainval.txt 

测试集： 2007/test.txt

训练:config文件修数据集路径之后直接运行train.py即可训练


自制的流程可视化图,更方便从宏观角度看待模型 见 https://www.yuque.com/pangkun/mizone/xhabah
