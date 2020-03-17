此代码源自https://github.com/chenyuntc/simple-faster-rcnn-pytorch

在此基础上进行修改与注释(但是在COCO或者VOC数据集上的精度我并没有测试)

陈云版本的nms以及roi_pooling部分我把它更换成python代码,并简化合并了很多代码,主要是为了便于理解Faster-RCNN的代码流程. 但是速度大概为他的三分之一

准备数据集:按照只要在Annotations和JPGImages分别放入xml和jpg文件,然后运行xml2txt.py即可生成相应的训练与验证数据.

训练:config文件配置好之后直接运行train.py即可训练(同时每个epoch都可以验证mAP)

###### 注:如果从头开始训练一个模型的话需要预先加载vgg16的官方权重作为特征提取.否则最终模型效果可能会不太理想.当然如果你已经训练出一个不错的Faster-RCNN模型的话可以忽略此步.因为后者会自动覆盖前者.

测试:dection_img.py文件为最终测试模型效果的文件,只需要提供一个待测模型及待检测图片文件夹即可运行