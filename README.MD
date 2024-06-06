此代码源自 https://github.com/chenyuntc/simple-faster-rcnn-pytorch

并在此基础上进行修改与注释.以下为基于VOC2007数据集的指标,(主干网络为vgg16)

|是否忽略困难样本 |mAP|
|---          |--- |
|是(simple实现)      |0.6944|
|否      |0.6213|


训练集： 2007/trainval.txt 

测试集： 2007/test.txt

训练:config文件修数据集路径之后直接运行train.py即可训练


自制的流程可视化图,更方便从宏观角度看待模型 见 https://www.yuque.com/pangkun/kb/lykze8

注:本份代码仅仅建议学习使用,实际应用应去寻求其他实现,比如 MMdetection或者Detectron2 

因为这些框架中的扩展性更强、参数调的更好.不管是学习还是实际应用都挺不错的.但是一开始上手可能会比较困难
