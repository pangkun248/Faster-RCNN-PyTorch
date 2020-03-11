此代码源自https://github.com/chenyuntc/simple-faster-rcnn-pytorch

在此基础上进行修改与注释(但是在COCO或者VOC数据集上的精度我并没有测试)

陈云版本的nms以及roi_pooling部分我把它更换成python代码,并简化合并了很多代码,主要是为了便于理解Faster-RCNN的代码流程.
但是速度大概为他的六分之一

训练:config文件配置好之后直接运行train即可训练(同时每个epoch都可以验证mAP)

测试:dection_img.py文件为最终测试模型效果的文件,只需要提供一个待测模型即可运行
