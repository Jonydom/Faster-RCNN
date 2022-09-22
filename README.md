# 基于MindSpore框架的Faster-RCNN案例实现

## 1 模型简介

Faster-RCNN模型于2016年在论文《Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks》中被提出，它的提出是为了改进Fast-RCNN区域候选算法运行时间长的问题。

Faster-RCNN模型是在Fast-RCNN模型的基础上建立的，由于目标检测网络依靠区域候选算法（如Selective Search）来假设目标的位置，运算时间开销很大，于是Faster-RCNN提出了一个可以共享卷积特征图的深度全卷积网络RPN来代替区域候选算法，使用RPN网络产生的候选区域进行分类与边框回归计算，从而大大加快了运行速度。

### 1.1 模型结构

Faster R-CNN是个两阶段的目标检测方法，主要由提取候选区域的全卷积网络RPN与Fast R-CNN检测器组成，整个检测过程通过一个网络完成。RPN和Fast R-CNN的配合作用可以理解为一种注意力机制，先大致确定目标在视野中的位置，然后再锁定目标仔细观察，确定目标的类别和更加准确的位置。 图1为论文中给出的Faster-RCNN基本结构图。

![image text](https://github.com/514forever/IMG/blob/main/%E5%9B%BE%E7%89%871.png)

Faster R-CNN检测部分主要可以分为以下四个模块：

（1）Conv layers特征提取网络，采用VGG16、ResNet等常用结构作为特征提取的模块，用于提取输入图像特征。然后将提取得到的图像特征feature maps用于后续的RPN层生成一系列可能的候选框。

（2）RPN候选检测框生成网络，该网络替代了之前Fast R-CNN版本的Selective Search，用于生成候选框proposal ，输出为一系列候选框以及每个候选框框中目标的概率值。

（3）RoI Pooling兴趣域池化，以RPN网络输出的兴趣区域和Conv layers输出的图像特征为输入，将两者进行综合后得到固定大小的区域特征图，后续将其送入全连接层继续做目标分类和坐标回归。

（4）Classification and Regression分类与回归。利用上一层得到的区域特征图通过softmax对图像进行分类，并通过边框回归修正物体的精确位置，输出兴趣区域中物体所属的类别以及物体在图像中精确的位置。

### 1.2 模型特点

使用RPN来生成候选区域，完全使用CNN解决目标检测任务，将特征提取、候选框选取、边框回归和分类都整合到一个网络中。

## 2 案例实现

### 2.1 环境准备与数据读取
![](https://cdn.jsdelivr.net/gh/Jonydom/myPic/img/1311.jpg)
![](https://cdn.jsdelivr.net/gh/Jonydom/myPic/img/30345.jpg)
### 2.2 数据集创建

### 2.3 模型构建

