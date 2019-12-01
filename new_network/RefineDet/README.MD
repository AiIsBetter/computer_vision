论文：Single-Shot Refinement Neural Network for Object Detection 论文链接：https://arxiv.org/abs/1711.06897 

SSD开源框架地址：https://github.com/pierluigiferrari/ssd_keras
（解释特别详细，适合没有学习过SSD的人配合论文认真研读代码）

RefineDet是发表在2018cvpr的目标检测论文， 在VOC2007、2012、COCO上取得了sota。git上面有作者编写的caffe版本和pytorch版本的开源代码。但是因为前期使用git开源的的keras版本的SSD框架源码进行的实验，考虑到现有标注数据读取方便，还是使用相同的框架来进行开发比较好，所以实现论文算法的方式是保留SSD框架的加载图片以及数据扩增以及其它一些传参的功能结构不动，其它所有算法相关部分重新进行了编写和修改。主要包括算法的网络结构，损失函数，解码、编码函数的编写。最终实现的效果，在我的数据集上面不使用imagenet等进行finetune，比SSD提高了5个点以上，速度下降在10%~20%之间。特别是对于小目标的检测好了很多，但是作者也提到小目标这方面还有提升的空间。

训练用的图像是320*320，训练用的.csv文件格式入下图所示：
![输入图片说明](https://images.gitee.com/uploads/images/2019/0108/113448_94d1e097_1805564.png "内网通截图20190108113342.png")

