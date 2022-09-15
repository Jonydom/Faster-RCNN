# 基于MindSpore框架的Faster-RCNN案例实现

## 1 模型简介

Faster-RCNN模型于2016年在论文《Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks》中被提出，它的提出是为了改进Fast-RCNN区域候选算法运行时间长的问题。

Faster-RCNN模型是在Fast-RCNN模型的基础上建立的，由于目标检测网络依靠区域候选算法（如Selective Search）来假设目标的位置，运算时间开销很大，于是Faster-RCNN提出了一个可以共享卷积特征图的深度全卷积网络RPN来代替区域候选算法，使用RPN网络产生的候选区域进行分类与边框回归计算，从而大大加快了运行速度。

### 1.1 模型结构

### 1.2 模型特点

## 2 案例实现

### 2.1 环境准备与数据读取

### 2.2 数据集创建

### 2.3 模型构建

