此代码源自https://github.com/chenyuntc/simple-faster-rcnn-pytorch

在此基础上进行修改与注释(但是在COCO或者VOC数据集上的精度我并没有测试)

陈云版本的nms使用的是cupy,我把它更换成cuda了,并简化合并了很多代码,主要是为了便于理解Faster-RCNN的代码流程.

但是速度大概为他的三分之一